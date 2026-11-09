from __future__ import division, print_function

# ==================== IMPORTS ====================
# Standard library imports
import os
import os.path as osp
import random
import argparse
import logging
import shutil

# Visualization and data processing
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

# Deep learning frameworks
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Machine learning utilities
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score

# Transformers and model training
from transformers import LlavaForConditionalGeneration, LlavaProcessor
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
import wandb

# Custom utilities
import tools.imutils as imutils
import tools.utils as utils
import tools.pyutils as pyutils
from tools.utils import log_sensitivity_at_specificity, compute_group_auc, ImprovedBalancedBatchSampler, log_multiclass_sensitivity_at_specificity
from data_loaders.labels import binary_label, filter_valid_samples

# ==================== CONSTANTS ====================
IMAGE_SIZE = 336

# Demographic group labels for bias analysis
DEMOGRAPHIC_GROUPS = {
    'Sex': ['0', '1'],
    'Race': ["Asian", "Black or African American", "White", "Other or Unknown"],
    'Ethnic': ["0", "1", "Unknown or Not Reported"],
    'Language': ["English", "Spanish", "Other or Unknown"]
}

# Task configuration mapping
TASK_CONFIG = {
    'dr': (-3, 'Diabetic Retinopathy'), 
    'amd': (-2, 'Aged Macular Degeneration'), 
    'glaucoma': (-1, 'Glaucoma')
}

# ==================== DATA PROCESSING FUNCTIONS ====================
def mask_until_after_assistant(labels: torch.Tensor, tokenizer, assistant_token_ids: list):
    """
    Mask tokens until after the assistant token for proper loss computation.
    
    Args:
        labels: Token labels tensor
        tokenizer: Model tokenizer
        assistant_token_ids: List of assistant token IDs
    
    Returns:
        Masked labels tensor
    """
    for i in range(labels.size(0)):
        for j in range(labels.size(1) - len(assistant_token_ids) + 1):
            if torch.equal(labels[i, j:j+len(assistant_token_ids)], 
                          torch.tensor(assistant_token_ids, device=labels.device)):
                labels[i, :j + len(assistant_token_ids)] = -100  # Mask until ASSISTANT:
                break
    return labels


def collate_fn(examples):
    """
    Custom collate function for processing batches of multimodal data.
    
    Args:
        examples: List of training examples
    
    Returns:
        Processed batch dictionary
    """
    # Process text templates
    texts = [processor.apply_chat_template(e, tokenize=False) for e in examples]
    
    # Process images
    images = []
    for example in examples:
        image = Image.open(example[1]['content'][0]["image"]).convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        images.append(image)

    # Tokenize and process batch
    batch = processor(
        text=texts, 
        images=images, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=768
    )
    
    # Create labels and apply masking
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels = mask_until_after_assistant(labels, processor.tokenizer, assistant_id)
    labels[:, -1] = -100
    batch["labels"] = labels
    
    return batch


def format_data(sample, task_idx, disease_name, system_message, img_root_path):
    """
    Format training data sample into chat template structure.
    
    Args:
        sample: Raw data sample
        task_idx: Task index for label extraction
        disease_name: Name of the disease
        system_message: System prompt message
        img_root_path: Root path for images
    
    Returns:
        Formatted chat conversation
    """
    task = {-3: "dr", -2: "amd", -1: "glaucoma"}[task_idx]
    label = "positive" if binary_label(sample[task_idx], task) else "negative"
    prompt = f"Please diagnose whether the {disease_name} exist or not based on the given image.\n"
    
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image", "image": os.path.join(img_root_path, sample[1])},
            {"type": "text", "text": prompt},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": str(label)}]}
    ]


def format_data_for_inference(sample, disease_name, system_message, img_root_path):
    """
    Format validation data sample for inference (without assistant response).
    
    Args:
        sample: Raw data sample
        disease_name: Name of the disease
        system_message: System prompt message
        img_root_path: Root path for images
    
    Returns:
        Formatted chat conversation for inference
    """
    prompt = f"Please diagnose whether the {disease_name} exist or not based on the given image."

    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image", "image": os.path.join(img_root_path, sample[1])},
            {"type": "text", "text": prompt}
        ]}
    ]


def create_balanced_subset(data, task_idx, train=True):
    task = {-3: "dr", -2: "amd", -1: "glaucoma"}[task_idx]
    valid_samples = filter_valid_samples(data, task)
    negative = [sample for sample in valid_samples if binary_label(sample[task_idx], task) == 0]
    positive = [sample for sample in valid_samples if binary_label(sample[task_idx], task) == 1]
    if not positive:
        raise ValueError(f"No positive {task} samples are available.")

    if train:
        negative_ratio = 10 if task == "amd" else 5
        negative_count = min(len(negative), negative_ratio * len(positive))
        return random.sample(negative, negative_count), positive
    if task == "amd":
        return random.sample(negative, min(len(negative), len(positive))), positive
    return negative, positive

# ==================== EVALUATION FUNCTIONS ====================
def slice_logits(logits, labels):
    """
    Extract and process logits for evaluation.
    
    Args:
        logits: Model output logits
        labels: Target labels
    
    Returns:
        Processed logits tensor
    """
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.detach().cpu()


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics from predictions.
    
    Args:
        eval_pred: Evaluation prediction object
    
    Returns:
        Dictionary of computed metrics
    """
    logits = torch.tensor(eval_pred.predictions)
    token_ids = logits.argmax(dim=-1)  # (B, L): predicted token at each position

    # Extract logits for positive/negative tokens
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
    
    # Calculate probabilities and ground truth
    probs = torch.sigmoid(logits[:, -1, POS_ID[0]] - logits[:, -1, NEG_ID[0]]).numpy()
    labels = torch.tensor(eval_pred.label_ids)
    gt_ids = labels[labels != -100].view(logits.size(0), -1)[:, 0]
    y_true = (gt_ids == POS_ID[0]).int().cpu().numpy()
    
    # Compute metrics
    auc_val = roc_auc_score(y_true, probs)
    fpr, tpr, thr = roc_curve(y_true, probs)
    best_threshold = thr[np.argmax(tpr - fpr)]
    accuracy = accuracy_score(y_true, probs >= best_threshold)
    
    return {"roc_auc": auc_val, "accuracy": accuracy}


def run_custom_evaluation(trainer, val_dataset, val_labels):
    """
    Run custom evaluation on validation dataset.
    
    Args:
        trainer: SFT trainer object
        val_dataset: Validation dataset
        val_labels: Validation labels
    
    Returns:
        Dictionary of evaluation metrics
    """
    outputs = trainer.predict(val_dataset)
    logits = torch.from_numpy(outputs.predictions)  # (B, S, L)
    probs = torch.sigmoid(logits[:, -1, POS_ID[0]] - logits[:, -1, NEG_ID[0]]).numpy()

    auc_val = roc_auc_score(val_labels, probs)
    print(f"[Custom Eval] AUC: {auc_val:.4f}")
    
    return {"auc": auc_val}


# ==================== MODEL SETUP FUNCTIONS ====================
def setup_model_and_processor(model_id):
    """
    Initialize and configure the LLaVA model and processor.
    
    Args:
        model_id: HuggingFace model identifier
    
    Returns:
        Tuple of (model, processor)
    """
    # Load model
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        device_map=None, 
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # Load processor and configure
    processor = LlavaProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "right"
    processor.image_processor.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    processor.image_processor.crop_size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}
    
    return model, processor


def setup_peft_config():
    """
    Setup LoRA (Parameter Efficient Fine-Tuning) configuration.
    
    Returns:
        PEFT configuration object
    """
    return LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )


def setup_training_config(exp_name, seed=0, report_to_wandb=False):
    """
    Setup training configuration for SFT.
    
    Args:
        exp_name: Experiment name
    
    Returns:
        SFT training configuration
    """
    training_args = SFTConfig(
        output_dir=exp_name,
        seed=seed,
        data_seed=seed,
        num_train_epochs=15,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=1e-3,
        lr_scheduler_type="constant",
        logging_steps=10,
        eval_steps=10000,
        save_steps=10000,
        eval_strategy="steps",
        save_strategy="epoch", 
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=False,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        push_to_hub=False,
        report_to="wandb" if report_to_wandb else "none",
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=512,
    )
    training_args.remove_unused_columns = False
    
    return training_args


def setup_logging(exp_name):
    """
    Setup logging configuration.
    
    Args:
        exp_name: Experiment name
    
    Returns:
        Configured logger
    """
    log_path = os.path.join(exp_name, "log.txt")
    
    if osp.isfile(log_path):
        os.remove(log_path)
   
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    logger.addHandler(file_handler)
    
    return logger


# ==================== INFERENCE FUNCTIONS ====================
def run_inference_evaluation(model, processor, val_dataset, val_dataset_group, task_idx, logger):
    """
    Run inference evaluation on validation dataset with demographic analysis.
    
    Args:
        model: Trained model
        processor: Model processor
        val_dataset: Validation dataset
        val_dataset_group: Demographic group information
        task_idx: Task index
        logger: Logger object
    """
    batch_size = 1
    model.eval()

    # Initialize result containers
    preds, targets = [], []
    infos = {'sex': [], 'race': [], 'ethnic': [], 'language': []}

    print("Starting inference evaluation...")
    
    for i in tqdm(range(0, len(val_dataset), batch_size), desc="Running inference"):
        batch = val_dataset[i:i + batch_size]
        batch_group = val_dataset_group[i:i + batch_size]

        # Prepare inputs
        texts, images = [], []
        for example in batch:
            text = processor.apply_chat_template(
                example, add_generation_prompt=True, tokenize=False
            ).strip()
            texts.append(text)
            
            image = Image.open(example[1]['content'][0]["image"]).convert("RGB")
            image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
            images.append([image])

        # Run inference
        with torch.no_grad():
            texts[0] += "\n"
            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True
            ).to(model.device)

            outputs = model(**inputs, output_hidden_states=False, return_dict=True)
            logits = outputs.logits

            # Calculate probability
            probs = torch.sigmoid(logits[0, -1, POS_ID] - logits[0, -1, NEG_ID])
            predicted_token = processor.tokenizer.decode(outputs.logits[0].argmax(-1)[-1])
            print(f"==> {predicted_token} | {probs}")

            # Process target and demographic info
            task = {-3: "dr", -2: "amd", -1: "glaucoma"}[task_idx]
            target = float(binary_label(batch_group[0][task_idx], task))
            
            demo_info = {
                'sex': str(batch_group[0][0]),
                'race': str(batch_group[0][1]),
                'ethnic': str(batch_group[0][2]),
                'language': str(batch_group[0][3]),
            }

            # Store results
            preds.append(probs.detach().cpu().item())
            targets.append(target)
            
            for key in infos.keys():
                infos[key].append(demo_info[key])

    # Convert to numpy arrays and compute metrics
    targets, preds = np.array(targets), np.array(preds)
    auc_score = roc_auc_score(targets, preds)
    logger.info(f"Overall AUC: {auc_score:.4f}")
    
    # Compute equality of opportunity metrics
    log_sensitivity_at_specificity(targets, preds, logger)
    
    # Compute group-wise AUC scores
    group_names = ['Sex', 'Race', 'Ethnic', 'Language']
    group_labels = list(DEMOGRAPHIC_GROUPS.values())
    
    for group, labels in zip(group_names, group_labels):
        compute_group_auc(
            targets, preds, infos[group.lower()], labels, 
            group, logger, auc_score, False
        )


# ==================== MAIN EXECUTION ====================
def main():
    """Main execution function."""
    # ==================== ARGUMENT PARSING ====================
    parser = argparse.ArgumentParser(description='LLaVA Medical Image Classification')
    parser.add_argument("--task", required=True, choices=['amd', 'dr', 'glaucoma'],
                       help='Medical task: amd, dr, or glaucoma')
    parser.add_argument("--name", required=True, help='Experiment name')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument(
        "--adapter-checkpoint",
        help="Optional trained LoRA adapter checkpoint to evaluate instead of training",
    )
    args = parser.parse_args()

    # ==================== SETUP ====================
    # Set random seeds for reproducibility
    pyutils.same_seeds(args.seed)
    cudnn.benchmark = False

    # Get task configuration
    task_idx, disease_name = TASK_CONFIG[args.task]
    
    # Create system message
    system_message = f"""You are an expert AI in ophthalmology.
Your primary role is to provide accurate, reliable, and up-to-date medical knowledge based on credible sources.
You must follow these guidelines:
1. Be accurate, concise, and clinically relevant.
2. Use proper medical terms.
3. Avoid overexplaining unless requested.
4. Tone: confident, professional, precise.
Do not include any explanation or thought.
If {disease_name} is present, answer exactly 'positive'. Otherwise answer 'negative'."""

    # ==================== DATA LOADING ====================
    print("Loading datasets...")
    img_root_path = 'data/images'
    train_dataset_raw = np.load('data/splits/train_final.npy', allow_pickle=True)
    val_dataset_raw = np.load('data/splits/val_final.npy', allow_pickle=True)

    print("Creating manuscript-balanced subsets...")
    train_dataset_raw = sum(create_balanced_subset(train_dataset_raw, task_idx, train=True), [])
    val_dataset_raw = sum(create_balanced_subset(val_dataset_raw, task_idx, train=False), [])

    # Format datasets
    print("Formatting training data...")
    train_dataset = [
        format_data(s, task_idx, disease_name, system_message, img_root_path) 
        for s in tqdm(train_dataset_raw)
    ]
    random.shuffle(train_dataset)

    print("Formatting validation data...")
    val_dataset = []
    val_dataset_group = []
    for s in tqdm(val_dataset_raw):
        val_dataset.append(format_data_for_inference(s, disease_name, system_message, img_root_path))
        val_dataset_group.append(s[2:])

    # val_labels = [1 if s[task_idx] != '0.0' else 0 for s in val_dataset_raw]

    print("=" * 50)
    print(f"Dataset sizes | Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print("=" * 50)

    # ==================== MODEL SETUP ====================
    global processor, POS_ID, NEG_ID, assistant_id  # Make these global for collate_fn
    
    print("Setting up model and processor...")
    model_id = "llava-hf/llava-1.5-7b-hf"
    model, processor = setup_model_and_processor(model_id)

    # Get token IDs
    POS_ID = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize("positive"))
    NEG_ID = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize("negative"))
    assistant_id = processor.tokenizer.convert_tokens_to_ids(processor.tokenizer.tokenize("ASSISTANT:"))

    # Setup PEFT configuration
    peft_config = setup_peft_config()

    # ==================== EXPERIMENT SETUP ====================
    exp_name = f"{model_id.split('/')[-1]}-{args.name}"

    if args.adapter_checkpoint:
        print("🔁 Loading trained PEFT weights...")
        model = PeftModel.from_pretrained(model, args.adapter_checkpoint)
        phase = "val"
    else:
        print("🚀 Initializing new LoRA model...")
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        phase = "train"

    # Setup training configuration
    training_args = setup_training_config(
        exp_name, seed=args.seed, report_to_wandb=args.wandb
    )

    if args.wandb:
        wandb.init(project=f"{exp_name}-Project", name=exp_name, config=training_args)

    # ==================== TRAINER SETUP ====================
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        peft_config=peft_config,
        processing_class=processor.tokenizer,
        preprocess_logits_for_metrics=slice_logits,
    )

    # Setup experiment directory and logging
    if not os.path.exists(exp_name):
        os.makedirs(exp_name, exist_ok=True)

    logger = setup_logging(exp_name)

    # Copy source code for reproducibility
    shutil.copy(
        "./train_llava_ft_final.py",
        os.path.join(".", exp_name, "train_llava_ft_final_copy.py")
    )

    # ==================== TRAINING ====================
    if phase == 'train':
        print("Starting training...")
        trainer.train()
        trainer.save_model(training_args.output_dir)
        print("Training completed!")

    # ==================== EVALUATION ====================
    print("Starting evaluation...")
    run_inference_evaluation(model, processor, val_dataset, val_dataset_group, task_idx, logger)
    print("Evaluation completed!")


if __name__ == '__main__':
    main()