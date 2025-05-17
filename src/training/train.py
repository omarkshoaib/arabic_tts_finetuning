#!/usr/bin/env python3
"""
Training script for OuteTTS v1.0 using Unsloth optimization.
"""

import os
import glob
import argparse
import yaml
import polars as pl
import torch
from tqdm import tqdm
from loguru import logger
from datasets import Dataset
from unsloth import FastModel

def load_dataset_from_parquet(folder: str) -> Dataset:
    """
    Load prompts stored as *.parquet files into a HuggingFace dataset.
    
    Args:
        folder: Directory containing the parquet files
        
    Returns:
        Dataset object
    """
    prompts = []
    logger.info(f"Loading parquet files from {folder}")
    parquet_files = glob.glob(os.path.join(folder, "*.parquet"))
    
    if not parquet_files:
        logger.error(f"No parquet files found in {folder}")
        raise ValueError(f"No parquet files found in {folder}")
    
    for fpath in tqdm(parquet_files, desc="Loading parquet files"):
        try:
            df = pl.read_parquet(fpath)
            prompts.extend(df["prompt"].to_list())
        except Exception as exc:
            logger.error(f"Error loading {fpath}: {exc}")
    
    logger.info(f"Loaded {len(prompts)} prompts from {len(parquet_files)} parquet files")
    return Dataset.from_dict({"text": prompts})

def train(
    data_dir: str,
    output_dir: str,
    model_name: str = "OuteAI/Llama-OuteTTS-1.0-1B",
    val_split: float = 0.1,
    num_epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 5e-5,
    lora_r: int = 128,
    lora_alpha: int = 128,
    lora_target_modules: list = None,
    max_seq_length: int = 2048,
    save_steps: int = 100,
    seed: int = 42,
):
    """
    Train the model using Unsloth optimization.
    
    Args:
        data_dir: Directory containing the training data (parquet files)
        output_dir: Directory to save the model
        model_name: Name or path of the model to fine-tune
        val_split: Fraction of data to use for validation
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        gradient_accumulation_steps: Number of steps to accumulate gradients
        learning_rate: Learning rate for training
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_target_modules: List of modules to apply LoRA to
        max_seq_length: Maximum sequence length
        save_steps: How often to save the model
        seed: Random seed for reproducibility
    """
    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "v_proj"]
    
    logger.info(f"Training model with the following parameters:")
    logger.info(f"Model: {model_name}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"LoRA rank: {lora_r}")
    logger.info(f"LoRA alpha: {lora_alpha}")
    logger.info(f"LoRA target modules: {lora_target_modules}")
    
    # Load model with Unsloth optimization
    logger.info("Loading model with Unsloth optimization...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float32,  # Must use fp32 for OuteTTS to avoid issues
        load_in_4bit=False,   # No quantization for OuteTTS
    )
    
    # Apply LoRA adapters
    logger.info("Applying LoRA adapters...")
    model = FastModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=lora_target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
        use_rslora=False,
    )
    
    # Load dataset
    logger.info("Loading dataset...")
    raw_dataset = load_dataset_from_parquet(data_dir)
    
    # Split dataset
    logger.info(f"Splitting dataset with validation split: {val_split}")
    dataset = raw_dataset.train_test_split(test_size=val_split, seed=seed)
    
    # Tokenize dataset
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
    
    logger.info("Tokenizing dataset...")
    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    # Create data collator
    from transformers import DataCollatorForLanguageModeling
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Setup training arguments
    from transformers import TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=save_steps,
        logging_steps=10,
        learning_rate=float(learning_rate),
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        report_to="none",
        save_total_limit=3,
        fp16=False,  # OuteTTS requires fp32
        seed=seed,
    )
    
    # Initialize trainer
    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        data_collator=collator,
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"Training complete. Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model saved successfully")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train OuteTTS v1.0 with Unsloth optimization")
    parser.add_argument("--config", type=str, help="Path to the config YAML file")
    parser.add_argument("--data_dir", type=str, help="Directory containing the training data (parquet files)")
    parser.add_argument("--output_dir", type=str, help="Directory to save the model")
    
    args = parser.parse_args()
    
    # Load config from file if provided, otherwise use command-line args
    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        
        # Override config with command-line args if provided
        if args.data_dir:
            config["data_dir"] = args.data_dir
        if args.output_dir:
            config["output_dir"] = args.output_dir
    else:
        if not args.data_dir or not args.output_dir:
            parser.error("Either --config or both --data_dir and --output_dir must be provided")
        
        config = {
            "data_dir": args.data_dir,
            "output_dir": args.output_dir,
            # Default values
            "model_name": "OuteAI/Llama-OuteTTS-1.0-1B",
            "val_split": 0.1,
            "num_epochs": 3,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 5e-5,
            "lora_r": 128,
            "lora_alpha": 128,
            "lora_target_modules": ["q_proj", "v_proj"],
            "max_seq_length": 2048,
            "save_steps": 100,
            "seed": 42,
        }
    
    # Create output directory if it doesn't exist
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Save config to output directory
    with open(os.path.join(config["output_dir"], "train_config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    # Train model
    train(
        data_dir=config["data_dir"],
        output_dir=config["output_dir"],
        model_name=config.get("model_name", "OuteAI/Llama-OuteTTS-1.0-1B"),
        val_split=float(config.get("val_split", 0.1)),
        num_epochs=int(config.get("num_epochs", 3)),
        batch_size=int(config.get("batch_size", 1)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 8)),
        learning_rate=float(config.get("learning_rate", 5e-5)),
        lora_r=int(config.get("lora_r", 128)),
        lora_alpha=int(config.get("lora_alpha", 128)),
        lora_target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"]),
        max_seq_length=int(config.get("max_seq_length", 2048)),
        save_steps=int(config.get("save_steps", 100)),
        seed=int(config.get("seed", 42)),
    )

if __name__ == "__main__":
    main() 