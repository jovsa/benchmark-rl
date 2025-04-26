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

from trl import PPOConfig, PPOTrainer, TextEnvironment


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
        default=32,
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


def parse_args(args=None):
    """Parse command line arguments."""
    parser = HfArgumentParser(ScriptArguments)
    if args is None:
        args = sys.argv[1:]
    script_args = parser.parse_args_into_dataclasses(args=args)[0]
    return script_args


def exact_match_reward(responses, answers=None):
    """Reward if generated response contains correct answer."""
    rewards = []
    pattern = r"Result\s*=\s*(-?\d+(?:\.\d+)?)\s*<submit>"
    for response, answer in zip(responses, answers):
        reward = 0.0
        try:
            predicted_number = None
            match_pattern = re.findall(pattern, response)
            if match_pattern:
                predicted_number = float(match_pattern[0])
            if predicted_number is not None:
                if np.abs(predicted_number - float(answer)) < 0.1:
                    reward += 1.0
        except Exception:
            pass
        rewards.append(torch.tensor(reward))
    return rewards


def evaluate(test_dataloader, text_env, ppo_trainer):
    test_rewards = []
    for test_batch in test_dataloader:
        _, _, _, rewards, _ = text_env.run(
            test_batch["query"],
            answers=test_batch["answer"],
        )
        test_rewards.extend(rewards)

    if not test_rewards:
        return torch.tensor(0.0)

    test_rewards = ppo_trainer.accelerator.gather_for_metrics(
        torch.stack(test_rewards).to(ppo_trainer.accelerator.device)
    )
    return test_rewards.mean()


def setup(script_args):
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        token=True,
    )
    # Wrap model in a way that TextEnvironment expects
    class WrappedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.pretrained_model = model
            self.device = model.device

        def __call__(self, *args, **kwargs):
            return self.pretrained_model(*args, **kwargs)

    wrapped_model = WrappedModel(model)
    ref_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        token=True,
    )
    wrapped_ref_model = WrappedModel(ref_model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name,
        token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

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
    ds = ds.select(range(1, len(ds)))  # skip first sample used in prompt

    ds_test = load_dataset("openai/gsm8k", "main", split="test")
    ds_test = ds_test.rename_columns({"question": "query"})
    ds_test = ds_test.map(lambda x: {"answer": x["answer"].split("#### ")[1]})

    # Tokenize datasets
    def tokenize(example):
        tokenized = tokenizer(text=example["query"])
        if tokenizer.eos_token_id is not None and tokenized["input_ids"][-1] != tokenizer.eos_token_id:
            tokenized["input_ids"] = tokenized["input_ids"] + [tokenizer.eos_token_id]
            tokenized["attention_mask"] = tokenized["attention_mask"] + [1]
        return tokenized

    ds = ds.map(tokenize, remove_columns="query")
    ds_test = ds_test.map(tokenize, remove_columns="query")

    return (
        wrapped_model,
        wrapped_ref_model,
        tokenizer,
        value_model,
        reward_model,
        ds,
        ds_test,
    )


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
        num_train_epochs=script_args.ppo_epochs,
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

    # Prepare test dataloader
    test_dataloader = ppo_trainer.accelerator.prepare(ds_test)

    # Setup text environment
    prompt = """\
Example of using a Python API to solve math questions.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

<request><PythonInterpreter>
def solution():
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result
print(solution())
<call>72<response>

Result = 72 <submit>

Q: """

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": -1,
        "max_new_tokens": script_args.max_new_tokens,
    }

    text_env = TextEnvironment(
        model=model,
        tokenizer=tokenizer,
        tools=[],  # No external tools needed
        reward_fn=exact_match_reward,
        prompt=prompt,
        max_turns=2,
        generation_kwargs=generation_kwargs,
    )

    # Training loop
    for epoch in range(script_args.n_epochs):
        for step, batch in enumerate(ppo_trainer.dataloader):
            if (step == 0) and (epoch % 4 == 0):  # evaluate every 4 epochs
                reward_mean_test = evaluate(
                    test_dataloader,
                    text_env,
                    ppo_trainer,
                )
            else:
                reward_mean_test = None

            queries, responses, masks, rewards, histories = text_env.run(
                batch["query"],
                answers=batch["answer"],
            )
            train_stats = ppo_trainer.step(queries, responses, rewards, masks)

            # Logging
            if reward_mean_test is not None:
                train_stats["env/reward_mean_test"] = reward_mean_test
            texts = {
                "query": batch["query"],
                "response": [
                    tokenizer.decode(response) for response in responses
                ],
                "answer": batch["answer"],
            }
            ppo_trainer.log_stats(
                train_stats,
                texts,
                rewards,
                columns_to_log=["query", "response", "answer"],
            )

    # Final evaluation
    reward_mean_test = evaluate(test_dataloader, text_env, ppo_trainer)
    ppo_trainer.save_pretrained(f"model/{script_args.model_name}-gsm8k")


if __name__ == "__main__":
    main()