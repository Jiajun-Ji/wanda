#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning script for BoolQ dataset (Boolean Question Answering).
Supports both sparse and dense models.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

# Import SparseTrainer from current directory
from sparse_trainer import SparseTrainer, check_sparsity

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to model checkpoint (pruned or dense)."},
    )
    use_flash_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use Flash Attention 3."},
    )
    use_sdpa: bool = field(
        default=False,
        metadata={"help": "Whether to use SDPA from PyTorch 2.0+."},
    )


@dataclass
class DataTrainingArguments:
    """Arguments for BoolQ data configuration."""
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of training samples."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of evaluation samples."},
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "Number of preprocessing workers."},
    )


def format_boolq_prompt(example):
    """
    Format BoolQ example into a prompt for causal LM.
    
    BoolQ format:
    - question: str
    - passage: str
    - answer: bool (True/False)
    
    Prompt format:
    Passage: {passage}
    Question: {question}
    Answer: {Yes/No}
    """
    passage = example['passage']
    question = example['question']
    answer = "Yes" if example['answer'] else "No"
    
    prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer: {answer}"
    return {"text": prompt}


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed
    set_seed(training_args.seed)

    # Load BoolQ dataset
    logger.info("Loading BoolQ dataset from super_glue...")
    raw_datasets = load_dataset("super_glue", "boolq")
    
    logger.info(f"Dataset loaded: {raw_datasets}")
    logger.info(f"Train samples: {len(raw_datasets['train'])}")
    logger.info(f"Validation samples: {len(raw_datasets['validation'])}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
    )
    
    # Set padding token if not exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Preprocess function
    def preprocess_function(examples):
        """
        Tokenize BoolQ examples using Llama-2 instruction format.

        Format:
        <s>[INST] 你会看到一个问题和一段文本。回答"True"或"False"。
        问题: {question}
        文本: {passage}
        答案: [/INST] {True|False}</s>
        """
        input_ids_list = []
        labels_list = []

        for i in range(len(examples['question'])):
            passage = examples['passage'][i]
            question = examples['question'][i]
            answer = "True" if examples['label'][i] else "False"

            # Llama-2 instruction format
            instruction = f"你会看到一个问题和一段文本。回答\"True\"或\"False\"。\n问题: {question}\n文本: {passage}\n答案:"

            # Full conversation
            full_text = f"<s>[INST] {instruction} [/INST] {answer}</s>"

            # Tokenize
            tokenized = tokenizer(
                full_text,
                max_length=data_args.max_length,
                truncation=True,
                padding=False,  # Don't pad yet
                add_special_tokens=False,  # We already added <s> and </s>
            )

            input_ids = tokenized["input_ids"]

            # Find the position of "[/INST]" to mask instruction part
            # Tokenize "[/INST]" to find its token IDs
            inst_end_tokens = tokenizer("[/INST]", add_special_tokens=False)["input_ids"]

            # Find where [/INST] ends in the sequence
            inst_end_pos = None
            for j in range(len(input_ids) - len(inst_end_tokens) + 1):
                if input_ids[j:j+len(inst_end_tokens)] == inst_end_tokens:
                    inst_end_pos = j + len(inst_end_tokens)
                    break

            # Create labels: mask instruction, keep answer
            labels = [-100] * len(input_ids)
            if inst_end_pos is not None:
                # Only compute loss on answer part (after [/INST])
                labels[inst_end_pos:] = input_ids[inst_end_pos:]

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        # Pad to max_length
        max_len = data_args.max_length
        padded_input_ids = []
        padded_labels = []

        for input_ids, labels in zip(input_ids_list, labels_list):
            # Truncate if too long
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]

            # Pad
            padding_length = max_len - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length

            padded_input_ids.append(input_ids)
            padded_labels.append(labels)

        return {
            "input_ids": padded_input_ids,
            "attention_mask": [[1 if token_id != tokenizer.pad_token_id else 0
                               for token_id in input_ids]
                              for input_ids in padded_input_ids],
            "labels": padded_labels,
        }

    # Preprocess datasets
    logger.info("Preprocessing datasets...")
    tokenized_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=raw_datasets["train"].column_names,
        desc="Tokenizing BoolQ dataset",
    )

    # Limit samples if specified
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(tokenized_datasets["train"]), data_args.max_train_samples)
        tokenized_datasets["train"] = tokenized_datasets["train"].select(range(max_train_samples))
    
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(tokenized_datasets["validation"]), data_args.max_eval_samples)
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(max_eval_samples))

    logger.info(f"Final train samples: {len(tokenized_datasets['train'])}")
    logger.info(f"Final eval samples: {len(tokenized_datasets['validation'])}")

    # Load model
    logger.info(f"Loading model from {model_args.model_name_or_path}...")
    
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    
    # Determine attention implementation
    attn_implementation = None
    if model_args.use_flash_attention:
        attn_implementation = "flash_attention_2"
        logger.info("Using Flash Attention 2")
    elif model_args.use_sdpa:
        attn_implementation = "sdpa"
        logger.info("Using SDPA")
    
    # Load model
    if is_distributed:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16,
            attn_implementation=attn_implementation,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation=attn_implementation,
        )

    # Check sparsity
    sparsity = check_sparsity(model)
    logger.info(f"Model sparsity: {sparsity:.2%}")

    # Initialize trainer
    trainer = SparseTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
    )

    # Training
    if training_args.do_train:
        logger.info("*** Training ***")
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        
        # Save metrics
        metrics = train_result.metrics
        metrics["train_samples"] = len(tokenized_datasets["train"])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        
        metrics["eval_samples"] = len(tokenized_datasets["validation"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("Training completed!")


if __name__ == "__main__":
    main()

