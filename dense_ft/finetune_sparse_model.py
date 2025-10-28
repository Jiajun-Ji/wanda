#!/usr/bin/env python
# coding=utf-8
"""
Full fine-tuning script for sparse pruned models using SparseTrainer.
This script trains ALL parameters while maintaining the sparsity pattern.
"""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
from datasets import load_dataset
import numpy as np

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# Import SparseTrainer from current directory
from sparse_trainer import SparseTrainer, check_sparsity

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pruned model checkpoint."},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path."},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for models."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use fast tokenizer."},
    )


@dataclass
class DataTrainingArguments:
    """Arguments for data configuration."""
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset name (e.g., 'wikitext', 'c4')."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset config (e.g., 'wikitext-2-raw-v1')."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of training samples."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of evaluation samples."},
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={"help": "Sequence length for training."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite cached datasets."}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={"help": "Validation split percentage."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "Number of preprocessing workers."},
    )


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
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {training_args}")

    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    # Set seed
    set_seed(training_args.seed)

    # Load dataset
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    if data_args.dataset_name == 'c4':
        raw_datasets = load_dataset(
            'allenai/c4', 'allenai--c4',
            data_files={
                'train': 'en/c4-train.00000-of-01024.json.gz',
                'validation': 'en/c4-validation.00000-of-00008.json.gz'
            }
        )
    else:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
        )

    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[:{data_args.validation_split_percentage}%]",
        )
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[{data_args.validation_split_percentage}%:]",
        )

    # Load config and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    # Load pruned model
    logger.info("Loading pruned model for full fine-tuning with SparseTrainer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        cache_dir=model_args.cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    # Check initial sparsity
    initial_sparsity = check_sparsity(model)
    logger.info(f"Initial model sparsity: {initial_sparsity:.4f}")

    # Make all parameters trainable
    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} || All params: {all_params:,} || Trainable%: {100 * trainable_params / all_params:.2f}")

    # Determine block size
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            block_size = 1024
    else:
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    logger.info(f"Using block size: {block_size}")

    # Tokenization and grouping functions
    def tokenize_function(examples):
        return tokenizer(examples["text"])

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Process datasets
    with training_args.main_process_first(desc="tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=raw_datasets["train"].column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    with training_args.main_process_first(desc="grouping"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Prepare train/eval datasets
    if training_args.do_train:
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Initialize SparseTrainer (key difference from regular Trainer!)
    trainer = SparseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )

    # Training
    if training_args.do_train:
        logger.info("*** Starting sparse full fine-tuning ***")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Check final sparsity
        final_sparsity = check_sparsity(model)
        logger.info(f"Final model sparsity: {final_sparsity:.4f}")
        logger.info(f"Sparsity maintained: {abs(final_sparsity - initial_sparsity) < 1e-6}")

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    logger.info("*** Sparse full fine-tuning complete! ***")


if __name__ == "__main__":
    main()

