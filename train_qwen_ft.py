from __future__ import division, print_function

# === Base ===
import os
import os.path as osp
import random
import argparse
import logging
from tqdm import tqdm
from matplotlib import pyplot as plt
import pdb
import shutil
import os

# === DL ===
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# === Custom ===
import tools.imutils as imutils
import tools.utils as utils
import tools.pyutils as pyutils
from tools.utils import log_sensitivity_at_specificity, compute_group_auc, ImprovedBalancedBatchSampler, log_multiclass_sensitivity_at_specificity
from data_loaders.labels import binary_label, filter_valid_samples

# === Evaluation ===
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score

# === Transformers ===
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, Qwen2VLImageProcessor
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import wandb

from qwen_vl_utils import process_vision_info

# === Label Masking Function ===
def keep_pos_neg(labels_1d):
    tgt_mask = torch.zeros_like(labels_1d, dtype=torch.bool)
    i = 0
    while i < len(labels_1d) - 2:
        if labels_1d[i] == im_start_id and labels_1d[i + 1] == assistant_id:
            j = i + 2
            while j < len(labels_1d) and labels_1d[j] != im_end_id:
                if labels_1d[j].item() in (POS_ID, NEG_ID):
                    tgt_mask[j] = True
                    break
                j += 1
            break
        i += 1
    masked = labels_1d.clone()
    masked[~tgt_mask] = -100
    return masked

# === Collate Function ===
def collate_fn(examples):
    texts = [processor.apply_chat_template(e, tokenize=False) for e in examples]
    images = [process_vision_info(e)[0] for e in examples]
    
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=512)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    image_tokens = [151652, 151653, 151655] if isinstance(processor, Qwen2VLProcessor) else [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for img_tok in image_tokens:
        labels[labels == img_tok] = -100

    for b in range(labels.size(0)):
        labels[b] = keep_pos_neg(labels[b])
    
    batch["labels"] = labels
    # print(labels[0])
    # import pdb; pdb.set_trace()
    return batch

# === Data Formatting ===
def format_data(sample):
    label = "positive" if binary_label(sample[task_idx], args.task) else "negative"
    prompt = f"Please diagnose whether the {disease_name} exist or not based on the given image."
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image", "image": os.path.join(img_root_path, sample[1]), "resized_height": 512, "resized_width": 512},
            {"type": "text", "text": prompt}
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": str(label)}]}
    ]

def format_data_for_inference(sample):
    prompt = f"Please diagnose whether the {disease_name} exist or not based on the given image. <|im_start|>assistant"
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image", "image": os.path.join(img_root_path, sample[1]), "resized_height": 512, "resized_width": 512},
            {"type": "text", "text": prompt}
        ]}
    ]

# === Logit Preprocessing ===
def slice_logits(logits, labels):
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.detach().cpu()

def build_single_inputs(dialog):
    """
    dialog: format_data_for_inference() 로 만든 2-turn dict
    반환: model.generate() 에 바로 넣을 수 있는 dict
    """
    # 1) 텍스트 템플릿
    text_prompt = processor.apply_chat_template(
        dialog, tokenize=False
    )                               # str (특수토큰 포함)

    # 2) 이미지 전처리
    image = process_vision_info(dialog)[0]      # PIL.Image or numpy.ndarray
                                                # (format_data_for_inference와 동일)
    # 3) 토크나이즈 + 이미지 인코딩
    encoded = processor(
        text=[text_prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)          # BatchEncoding(dict) → .to() 가능
    return encoded

# def slice_logits(logits, labels):
#     if isinstance(logits, (tuple, list)):
#         logits = logits[0]
#     pos_list, neg_list = [], []
#     for l, y in zip(logits, labels):
#         idx = next((j for j, tok in enumerate(y) if tok in (POS_ID, NEG_ID)), None)
#         if idx is None:
#             raise ValueError("POS/NEG token not found in sequence.")
#         pos_list.append(l[idx, POS_ID].float().cpu())
#         neg_list.append(l[idx, NEG_ID].float().cpu())
#     return torch.stack([torch.stack(pos_list), torch.stack(neg_list)], dim=1)

# === Metrics ===
def compute_metrics(eval_pred):
    logits = torch.tensor(eval_pred.predictions)
    probs = torch.sigmoid(logits[:, 1] - logits[:, 0]).numpy()
    labels = torch.tensor(eval_pred.label_ids)
    gt_ids = labels[labels != -100].view(logits.size(0), -1)[:, 0]
    y_true = (gt_ids == POS_ID).int().cpu().numpy()
    auc_val = roc_auc_score(y_true, probs)
    fpr, tpr, thr = roc_curve(y_true, probs)
    best = thr[np.argmax(tpr - fpr)]
    acc = accuracy_score(y_true, probs >= best)
    return {"roc_auc": auc_val, "accuracy": acc}

def run_custom_evaluation(trainer, val_dataset, val_labels):
    outputs = trainer.predict(val_dataset)
    logits = torch.from_numpy(outputs.predictions)  # (B, S, L)
    # pdb.set_trace()
    probs = torch.sigmoid(logits[:,-1, POS_ID] - logits[:,-1, NEG_ID]).numpy()

    # decoded = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # y_pred = [1 if "positive" in t.lower() else 0 for t in decoded]

    auc_val = roc_auc_score(val_labels, probs)
    # acc = accuracy_score(val_labels, y_pred)
    print(f"[Custom Eval] AUC: {auc_val:.4f}")
    # print(f"[Custom Eval] AUC: {auc_val:.4f}, ACC: {acc:.4f}")
    return {"auc": auc_val}

# def run_custom_evaluation(model, processor, dialogs, val_labels):
#     generations = []
#     for dialog in tqdm(dialogs):
#         inputs = build_single_inputs(dialog)   # ← 위 함수 사용

#         with torch.no_grad():
#             out_ids = model.generate(
#                 **inputs,
#                 max_new_tokens=4,
#                 return_dict_in_generate=True,
#                 output_scores=True,
#                 do_sample=False,
#                 # num_beams=1,
#             )
#         ans = processor.tokenizer.decode(out_ids.sequences[0][inputs.input_ids.size(1):], skip_special_tokens=True)
#         print(ans)
#         pdb.set_trace()
#         logits = out_ids.scores[0]
#         pos_logit = logits[:, POS_ID]
#         neg_logit = logits[:, NEG_ID]
#         probs = torch.sigmoid(pos_logit - neg_logit)

#         generations.append(probs.detach().cpu().numpy())

# #     y_pred = [1 if "positive" in g.lower() else 0 for g in generations]

#     # pdb.set_trace()

#     auc = roc_auc_score(val_labels, generations)
#     # acc = accuracy_score(val_labels, generations)
#     # print(f"[Custom Eval] AUC: {auc:.4f}, ACC: {acc:.4f}")
#     print(f"[Custom Eval] AUC: {auc:.4f}")
#     # return {"auc": auc, "accuracy": acc}
#     return {"auc": auc}



# === Main ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["amd", "dr", "glaucoma"])
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_subset", action='store_true')
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--adapter-checkpoint",
        help="Optional trained LoRA adapter checkpoint to evaluate instead of training",
    )
    args = parser.parse_args()

    pyutils.same_seeds(args.seed)

    task_map = {'dr': (-3, 'Diabetic Retinopathy'), 'amd': (-2, 'Aged Macular Degeneration'), 'glaucoma': (-1, 'Glaucoma')}
    task_idx, disease_name = task_map[args.task]
    # system_message = f"""You are an expert AI in ophthalmology.\nIf {disease_name} is present, answer exactly 'positive'. Otherwise answer 'negative'."""
    system_message = f"""You are an expert AI in ophthalmology.\n 
    Your primary role is to provide accurate, reliable, and up-to-date medical knowledge based on credible sources.\n
    "You must follow these guidelines:\n"
    "1. Be accurate, concise, and clinically relevant.\n"
    "2. Use proper medical terms.\n"
    "3. Avoid overexplaining unless requested.\n"
    "4. Tone: confident, professional, precise.\n"
    "Do not include any explanation or thought."
    If {disease_name} is present, answer exactly 'positive'. Otherwise answer 'negative'."""

    cudnn.benchmark = False
    img_root_path = 'data/images'
    train_dataset = np.load('data/splits/train_final.npy', allow_pickle=True)
    # val_dataset = np.load('data/splits/val_final.npy', allow_pickle=True)
    val_dataset_raw = np.load('data/splits/val_final.npy', allow_pickle=True)
    train_dataset = filter_valid_samples(train_dataset, args.task)
    val_dataset_raw = filter_valid_samples(val_dataset_raw, args.task)

    if args.use_subset:
        def subset(data,train=True):
            neg = [s for s in data if binary_label(s[task_idx], args.task) == 0]
            pos = [s for s in data if binary_label(s[task_idx], args.task) == 1]
            num_sample = len(pos)
            if train:
                return random.sample(neg, min(len(neg), 6 * num_sample)), pos
            else:
                return random.sample(neg, min(len(neg), num_sample)), pos
        train_dataset = sum(subset(train_dataset,train=True), [])
        val_dataset_raw = sum(subset(val_dataset_raw,train=False), [])

    train_dataset = [format_data(s) for s in tqdm(train_dataset)]
    random.shuffle(train_dataset)
    val_dataset = [format_data_for_inference(s) for s in tqdm(val_dataset_raw)]
    val_labels = [binary_label(s[task_idx], args.task) for s in val_dataset_raw]

    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
    processor = Qwen2VLProcessor.from_pretrained(model_id)
    processor.image_processor = Qwen2VLImageProcessor.from_pretrained(model_id, size={"height": 512, "width": 512}, crop_size={"height": 512, "width": 512})

    POS_ID = processor.tokenizer.convert_tokens_to_ids("positive")
    NEG_ID = processor.tokenizer.convert_tokens_to_ids("negative")
    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")#151644
    im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>") #151645
    assistant_id = processor.tokenizer.convert_tokens_to_ids("assistant")

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )


    exp_name = f"{model_id.split('/')[-1]}-{args.name}"

    if args.adapter_checkpoint:
        from peft import PeftModel
        print("🔁 Loading trained PEFT weights...")
        model = PeftModel.from_pretrained(model, args.adapter_checkpoint)
        phase= "val"
    else:
        print("🚀 Initializing new LoRA model...")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        phase= "train"

    training_args = SFTConfig(
        output_dir=exp_name,
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=10,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=1e-3,
        lr_scheduler_type="constant",
        # lr_scheduler_type="cosine",
        logging_steps=10,
        eval_steps=1000,
        save_steps=1000,
        eval_strategy="steps",
        save_strategy="steps",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        push_to_hub=False,
        report_to="wandb" if args.wandb else "none",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=512,
    )
    training_args.remove_unused_columns = False

    if args.wandb:
        wandb.init(project=f"{exp_name}-Project", name=exp_name, config=training_args)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        processing_class=processor.tokenizer,
        # compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=slice_logits,
    )

    os.makedirs(exp_name, exist_ok=True)
    shutil.copy("./train_qwen_ft.py",os.path.join(".",exp_name,"train_qwen_ft_copy.py"))

    if phase == 'train':
        trainer.train()
        trainer.save_model(training_args.output_dir)

    custom_eval_metrics = run_custom_evaluation(trainer, val_dataset, val_labels)
    # custom_eval_metrics = run_custom_evaluation(model, processor, val_dataset, val_labels)

