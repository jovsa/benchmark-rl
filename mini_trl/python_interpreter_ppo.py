import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from .trl_shim import PPOConfig, PPOTrainer
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(
        default="gpt2",
        metadata={"help": "the model name"},
    )
    learning_rate: Optional[float] = field(
        default=1e-5,
        metadata={"help": "the learning rate"},
    )
    mini_batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "the PPO minibatch size"},
    )
    batch_size: Optional[int] = field(
        default=16,
        metadata={"help": "the batch size"},
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16,
        metadata={"help": "the number of gradient accumulation steps"},
    )
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "max number of generated tokens per turn"},
    )
    ppo_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "max number of ppo epochs"},
    )
    n_epochs: Optional[int] = field(
        default=32,
        metadata={"help": "max number of ppo epochs"},
    )
    train_dataset_size: Optional[int] = field(
        default=100,
        metadata={"help": "number of training samples to use"},
    )


def parse_args(args=None):
    """Parse command line arguments."""
    parser = HfArgumentParser(ScriptArguments)
    if args is None:
        args = sys.argv[1:]
    script_args = parser.parse_args_into_dataclasses(args=args)[0]
    return script_args


def setup(script_args):
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        token=True,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        token=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name,
        token=True,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    # Load reward and value models
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels=1,
        token=True,
    )
    value_model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        num_labels=1,
        token=True,
    )

    # Load and prepare dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.rename_columns({"question": "query"})
    ds = ds.map(lambda x: {"answer": x["answer"].split("#### ")[1]})
    ds = ds.select(range(1, script_args.train_dataset_size + 1))  # use specified number of samples

    ds_test = load_dataset("openai/gsm8k", "main", split="test")
    ds_test = ds_test.rename_columns({"question": "query"})
    ds_test = ds_test.map(lambda x: {"answer": x["answer"].split("#### ")[1]})

    # Tokenize datasets
    def tokenize(example):
        # Tokenize input
        input_ids = tokenizer(
            text=example["query"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None,
        )

        # Convert answer to float, removing commas
        answer = float(example["answer"].replace(",", ""))

        return {
            "input_ids": input_ids["input_ids"],
            "attention_mask": input_ids["attention_mask"],
            "answer": answer,
        }

    ds = ds.map(tokenize, remove_columns=ds.column_names)
    ds_test = ds_test.map(tokenize, remove_columns=ds_test.column_names)

    # Set format for PyTorch
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "answer"])
    ds_test.set_format(type="torch", columns=["input_ids", "attention_mask", "answer"])

    return (
        model,
        ref_model,
        tokenizer,
        value_model,
        reward_model,
        ds,
        ds_test,
    )


def evaluate_model(model, tokenizer, dataset, batch_size=8):
    """Evaluate model on a dataset and return metrics."""
    model.eval()
    device = model.device
    total_reward = 0
    correct_predictions = 0
    total_samples = 0

    # Convert dataset to list for easier batching
    dataset_list = list(dataset)

    for i in range(0, len(dataset_list), batch_size):
        batch = dataset_list[i:i + batch_size]

        # Generate responses using input_ids
        responses = []
        answers = []
        for item in batch:
            # Decode input_ids to get the original query
            query = tokenizer.decode(item["input_ids"], skip_special_tokens=True)
            answers.append(item["answer"])

            # Generate response
            inputs = {
                "input_ids": item["input_ids"].unsqueeze(0).to(device),
                "attention_mask": item["attention_mask"].unsqueeze(0).to(device)
            }
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)

        # Compute rewards
        rewards = compute_rewards(responses, answers)
        total_reward += sum(rewards)

        # Count correct predictions
        for response, answer in zip(responses, answers):
            try:
                predicted_number = None
                match_pattern = re.findall(r"Result\s*=\s*(-?\d+(?:\.\d+)?)\s*<submit>", response)
                if match_pattern:
                    predicted_number = float(match_pattern[0])
                if predicted_number is not None and abs(predicted_number - float(answer)) < 0.1:
                    correct_predictions += 1
            except Exception:
                pass

        total_samples += len(batch)

    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    mean_reward = total_reward / total_samples if total_samples > 0 else 0

    return {
        "accuracy": accuracy,
        "mean_reward": mean_reward,
        "total_samples": total_samples
    }


def main(script_args=None):
    # Parse arguments if not provided
    if script_args is None:
        script_args = parse_args()

    # Setup models and data
    (
        model,
        ref_model,
        tokenizer,
        value_model,
        reward_model,
        ds,
        ds_test,
    ) = setup(script_args)

    # Configure PEFT
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_proj", "c_attn", "q_attn"],
    )

    # Configure PPO
    ppo_config = PPOConfig(
        output_dir="model",
        per_device_train_batch_size=script_args.batch_size,
        per_device_eval_batch_size=script_args.batch_size // 2,
        num_ppo_epochs=script_args.ppo_epochs,
        report_to="none",
    )

    # Create trainer
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        processing_class=tokenizer,
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=ds,
        eval_dataset=ds_test,
        peft_config=lora_config,
    )

    # dataset statistics
    print(f"Training dataset size: {len(ds)}")
    print(f"Test dataset size: {len(ds_test)}")

    # Evaluate before training
    print("\nPre-training evaluation:")
    train_metrics = evaluate_model(model, tokenizer, ds)
    test_metrics = evaluate_model(model, tokenizer, ds_test)
    print(f"Training set - Accuracy: {train_metrics['accuracy']:.4f}, Mean Reward: {train_metrics['mean_reward']:.4f}")
    print(f"Test set - Accuracy: {test_metrics['accuracy']:.4f}, Mean Reward: {test_metrics['mean_reward']:.4f}")

    # Train the model
    ppo_trainer.train()

    # Evaluate after training
    print("\nPost-training evaluation:")
    train_metrics = evaluate_model(model, tokenizer, ds)
    test_metrics = evaluate_model(model, tokenizer, ds_test)
    print(f"Training set - Accuracy: {train_metrics['accuracy']:.4f}, Mean Reward: {train_metrics['mean_reward']:.4f}")
    print(f"Test set - Accuracy: {test_metrics['accuracy']:.4f}, Mean Reward: {test_metrics['mean_reward']:.4f}")

    # Save the model
    model.save_pretrained(f"model/{script_args.model_name}-gsm8k")
    tokenizer.save_pretrained(f"model/{script_args.model_name}-gsm8k")


def compute_rewards(responses, answers):
    """Compute rewards for generated responses."""
    rewards = []
    for response, answer in zip(responses, answers):
        reward = 0.0
        try:
            predicted_number = None
            match_pattern = re.findall(r"Result\s*=\s*(-?\d+(?:\.\d+)?)\s*<submit>", response)
            if match_pattern:
                predicted_number = float(match_pattern[0])
            if predicted_number is not None:
                if np.abs(predicted_number - float(answer)) < 0.1:
                    reward += 1.0
        except Exception:
            pass
        rewards.append(torch.tensor(reward))
    return rewards


if __name__ == "__main__":
    main()