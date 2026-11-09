from __future__ import division, print_function


# medqa gsm8k openbookqa bioasq pubmedqa squad_v2

# === Base ===
import os
import os.path as osp
import random
import argparse
import logging
from tqdm import tqdm
from matplotlib import pyplot as plt
import pdb
from PIL import Image
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
from transformers import  AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, pipeline, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import wandb

# === Label Masking Function ===
def mask_until_after_assistant(labels: torch.Tensor, tokenizer, assistant_token_ids: list):
    for i in range(labels.size(0)):
        for j in range(labels.size(1) - len(assistant_token_ids) + 1):
            if torch.equal(labels[i, j:j+len(assistant_token_ids)], torch.tensor(assistant_token_ids, device=labels.device)):
                labels[i, :j + len(assistant_token_ids)] = -100  # ASSISTANT: 까지 마스킹
                break
    return labels


# === Collate Function ===
def load_image(image_path):
    with Image.open(image_path) as image:
        return image.convert("RGB").resize((IM_SIZE, IM_SIZE))


def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        image = load_image(example["image"])
        images.append([image])
        texts.append(processor.apply_chat_template(
            example["messages"], add_generation_prompt=False, tokenize=False
        ).strip())

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, with the padding and image tokens masked in
    # the loss computation
    labels = batch["input_ids"].clone()

    # Mask image tokens
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    # Mask tokens that are not used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    labels = mask_until_after_assistant(labels, processor.tokenizer, ASST_ID)
    labels[:,-1] = -100

    batch["labels"] = labels
    # pdb.set_trace()
    return batch

def format_data(sample):
    target = binary_label(sample[task_idx], args.task)
    label = "positive" if target else "negative"
    prompt = f"Please diagnose whether the {disease_name} exist or not based on the given image.\n"
    
    # pdb.set_trace()
    example = {}
    example["image"] = os.path.join(img_root_path, sample[1])
    example["label"] = target
    example["messages"] = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            # {"type": "image", "image": os.path.join(img_root_path, sample[1])},
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": str(label)}]}
    ]
    
    return example

def format_data_for_inference(sample):
    prompt = f"Please diagnose whether the {disease_name} exist or not based on the given image.\n"
    
    # pdb.set_trace()
    example = {}
    example["image"] = os.path.join(img_root_path, sample[1])
    # example["label"] = 0 if sample[task_idx]== '0,0' else 1
    example["messages"] = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            # {"type": "image", "image": os.path.join(img_root_path, sample[1])},
            {"type": "image"},
            {"type": "text", "text": prompt+"\n"},
        ]},
        # {"role": "assistant", "content": [{"type": "text", "text": str(label)}]}
    ]
    # prompt = f"Please diagnose whether the {disease_name} exist or not based on the given image."
    # return [
    #     {"role": "system", "content": [{"type": "text", "text": system_message}]},
    #     {"role": "user", "content": [
    #         {"type": "image", "image": os.path.join(img_root_path, sample[1])},
    #         {"type": "text", "text": prompt}
    #     ]}
    # ]
    return example

# === Logit Preprocessing ===
def slice_logits(logits, labels):
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.detach().cpu()
    
def compute_metrics(eval_pred):
    logits = torch.tensor(eval_pred.predictions)

    token_ids = logits.argmax(dim=-1)  # (B, L): predicted token at each position

    batch_logits = []
    for b in range(logits.size(0)):
        seq = token_ids[b]  # (L,)
        idxs = torch.where((seq == POS_ID[0]) | (seq == NEG_ID[0]))[0]
        if len(idxs) == 0:
            raise ValueError(f"Neither pos_id nor neg_id found in sequence {b}")
        t = idxs[0].item()  # first position where pos or neg appears
        tok_id = seq[t].item()  # should be either pos_id or neg_id
        batch_logits.append(logits[b, t, tok_id])  # scalar

    batch_logits = torch.stack(batch_logits)  # shape: [B]
    pred_texts = processor.tokenizer.batch_decode(token_ids[:,-1], skip_special_tokens=True)

    # print(pred_texts)
    # pdb.set_trace()
    probs = torch.sigmoid(logits[:,-1, POS_ID[0]] - logits[:,-1, NEG_ID[0]]).numpy()

    # probs = torch.sigmoid(batch_logits).numpy()
    labels = torch.tensor(eval_pred.label_ids)
    gt_ids = labels[labels != -100].view(logits.size(0), -1)[:, 0]
    y_true = (gt_ids == POS_ID[0]).int().cpu().numpy()
    auc_val = roc_auc_score(y_true, probs)
    fpr, tpr, thr = roc_curve(y_true, probs)
    best = thr[np.argmax(tpr - fpr)]
    acc = accuracy_score(y_true, probs >= best)
    return {"roc_auc": auc_val, "accuracy": acc}

def run_custom_evaluation(trainer, val_dataset, val_labels):
    outputs = trainer.predict(val_dataset)
    logits = torch.from_numpy(outputs.predictions)  # (B, S, L)
    # pdb.set_trace()
    probs = torch.sigmoid(logits[:,-1, POS_ID[0]] - logits[:,-1, NEG_ID[0]]).numpy()

    # decoded = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # y_pred = [1 if "positive" in t.lower() else 0 for t in decoded]

    auc_val = roc_auc_score(val_labels, probs)
    # acc = accuracy_score(val_labels, y_pred)
    print(f"[Custom Eval] AUC: {auc_val:.4f}")
    # print(f"[Custom Eval] AUC: {auc_val:.4f}, ACC: {acc:.4f}")
    return {"auc": auc_val}

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
    system_message = f"""You are an expert AI in ophthalmology.\n 
    Your primary role is to provide accurate, reliable, and up-to-date medical knowledge based on credible sources.\n
    You must follow these guidelines:\n
    1. Be accurate, concise, and clinically relevant.\n
    2. Use proper medical terms.\n
    3. Avoid overexplaining unless requested.\n
    4. Tone: confident, professional, precise.\n
    Do not include any explanation or thought.\n
    If {disease_name} is present, answer exactly 'positive'. Otherwise answer 'negative'."""
    # Diabetic Retinopathy (DR) is a diabetes-related eye disease that affects the retina — the light-sensitive tissue at the back of the eye. It occurs when chronically high blood sugar levels damage the small blood vessels in the retina, leading to leakage, blockage, or abnormal blood vessel growth.\n

    cudnn.benchmark = False
    img_root_path = 'data/images'
    train_dataset = np.load('data/splits/train_final.npy', allow_pickle=True)
    val_dataset_raw = np.load('data/splits/val_final.npy', allow_pickle=True)
    train_dataset = filter_valid_samples(train_dataset, args.task)
    val_dataset_raw = filter_valid_samples(val_dataset_raw, args.task)

    if args.use_subset:
        def subset(data,train=True):
            neg = [s for s in data if binary_label(s[task_idx], args.task) == 0]
            pos = [s for s in data if binary_label(s[task_idx], args.task) == 1]
            num_sample = len(pos)
            if train:
                return random.sample(neg, min(len(neg), 5 * num_sample)), pos
            else:
                return random.sample(neg, min(len(neg), num_sample)), pos
        train_dataset = sum(subset(train_dataset,train=True), [])
        val_dataset_raw = sum(subset(val_dataset_raw,train=False), [])

    train_dataset = [format_data(s) for s in tqdm(train_dataset)]
    random.shuffle(train_dataset)
    val_dataset = [format_data_for_inference(s) for s in tqdm(val_dataset_raw)]
    val_labels = [binary_label(s[task_idx], args.task) for s in val_dataset_raw]
    # val_dataset = [format_data(s) for s in tqdm(val_dataset)]
    print("="*50)
    print(f"Total number of Data| Train: {len(train_dataset)} | Val : {len(val_dataset)}")
    print("="*50)

    # model_id = "google/medgemma-4b-it"
    model_id = "google/medgemma-27b-it"
    model_kwargs = dict(
    attn_implementation="eager",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    )

    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        **model_kwargs
        # torch_dtype=torch.bfloat16,
        # device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # Use right padding to avoid issues during training
    processor.tokenizer.padding_side = "right"

    POS_ID = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize("positive")) #30558
    NEG_ID = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize("negative")) #27851
    ASST_ID = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize("model\n"))

    IM_SIZE = 256

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules="all-linear",
        # target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )


    exp_name = f"{model_id.split('/')[-1]}-{args.name}"

    if args.adapter_checkpoint:
        from peft import PeftModel
        print("🔁 Loading trained PEFT weights...")
        model = PeftModel.from_pretrained(model, args.adapter_checkpoint)
        phase= "val"
    else:
        print("🚀 Initializing new LoRA model...")
        # model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        phase= "train"
        

    training_args = SFTConfig(
        output_dir=exp_name,
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs= 12,                       # Number of training epochs
        per_device_train_batch_size=2,                           # Batch size per device during training
        per_device_eval_batch_size=4,                            # Batch size per device during evaluation
        gradient_accumulation_steps=8,                           # Number of steps before performing a backward/update pass
        gradient_checkpointing=True,                             # Enable gradient checkpointing to reduce memory usage
        optim="adamw_torch_fused",                               # Use fused AdamW optimizer for better performance
        logging_steps=10,                                        # Number of steps between logs
        save_strategy="epoch",                                   # Save checkpoint every epoch
        eval_strategy="steps",                                   # Evaluate every `eval_steps`
        eval_steps=10000,                                           # Number of steps between evaluations
        learning_rate=3e-4,                             # Learning rate based on QLoRA paper
        bf16=True,                                               # Use bfloat16 precision
        max_grad_norm=0.3,                                       # Max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                                       # Warmup ratio based on QLoRA paper
        lr_scheduler_type="linear",                              # Use linear learning rate scheduler
        # lr_scheduler_type="constant",                              # Use linear learning rate scheduler
        push_to_hub=False,
        report_to="wandb" if args.wandb else "tensorboard",
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Set gradient checkpointing to non-reentrant to avoid issues
        dataset_kwargs={"skip_prepare_dataset": True},           # Skip default dataset preparation to preprocess manually
        remove_unused_columns = False,                           # Columns are unused for training but needed for data collator
        label_names=["labels"],      
    )
    # training_args.remove_unused_columns = False

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
        # preprocess_logits_for_metrics=slice_logits,
    )
    
    os.makedirs(exp_name, exist_ok=True)
    shutil.copy("./train_medgemma_ft2.py",os.path.join(".",exp_name,"train_medgemma_ft2_copy.py"))

    if phase == 'train':
        trainer.train()
        trainer.save_model(training_args.output_dir)


    batch_size = 1
    model.eval()
    all_logits = []

    for i in tqdm(range(0, len(val_dataset), batch_size), desc="Running inference with logits"):
        batch = val_dataset[i:i + batch_size]

        # prepare inputs
        texts = []
        images = []
        for example in batch:
            text = processor.apply_chat_template(
                example["messages"], add_generation_prompt=True, tokenize=False
            ).strip()
            texts.append(text)
            image = load_image(example["image"])
            images.append([image])

        # tokenizer & image processor
        with torch.no_grad():
            texts[0] += "\n"
            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(model.device)

            outputs = model(**inputs, output_hidden_states=False, return_dict=True)

            print("==> ",processor.tokenizer.decode(outputs.logits[0].argmax(-1)[-1]))

            logits = outputs.logits
            # pdb.set_trace()
            probs = torch.sigmoid(logits[0,-1, POS_ID] - logits[0,-1, NEG_ID])
            # logits: (B, L, V)
            # all_logits.append(outputs.logits.to(torch.float32).detach().cpu().numpy())
            all_logits.append(probs)

    # pdb.set_trace()

    probs_all = torch.stack(all_logits,dim=0)
    probs_all = [prob.to(torch.float32).detach().cpu() for prob in probs_all]
    # logits= torch.from_numpy(np.stack(all_logits,axis=0)).squeeze(1)

    # probs = torch.sigmoid(logits[:,-1, POS_ID] - logits[:,-1, NEG_ID])

    # decoded = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # y_pred = [1 if "positive" in t.lower() else 0 for t in decoded]
    # pdb.set_trace()
    auc_val = roc_auc_score(val_labels, probs_all)
    print(auc_val)

    # print(trainer.evaluate())