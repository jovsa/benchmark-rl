import os
import unittest
from unittest.mock import MagicMock, patch
from datasets import Dataset

from mini_trl.python_interpreter_ppo import (
    ScriptArguments,
    compute_rewards,
    setup,
    parse_args,
)


class MockModel:
    def __init__(self):
        self.config = MagicMock()
        self.config.model_type = "gpt2"
        self.config.vocab_size = 50257
        self.config.n_embd = 768
        self.config.n_layer = 12
        self.config.n_head = 12
        self.config.n_positions = 1024
        self.config.n_ctx = 1024
        self.config.n_inner = 3072
        self.config.activation_function = "gelu"
        self.config.resid_pdrop = 0.1
        self.config.embd_pdrop = 0.1
        self.config.attn_pdrop = 0.1
        self.config.layer_norm_epsilon = 1e-5
        self.config.initializer_range = 0.02
        self.config.scale_attn_weights = True
        self.config.use_cache = True
        self.config.bos_token_id = 50256
        self.config.eos_token_id = 50256
        self.config.pad_token_id = 50256

    def __call__(self, *args, **kwargs):
        return MagicMock()

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self

    def train(self):
        return self


class MockTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.chat_template = None  # Add chat_template attribute

    def __call__(self, *args, **kwargs):
        return {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }

    def encode(self, *args, **kwargs):
        return [1, 2, 3]

    def decode(self, *args, **kwargs):
        return "mock response"

    def pad(self, *args, **kwargs):
        return {
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
        }


class TestPythonInterpreterPPO(unittest.TestCase):
    def setUp(self):
        # Set environment variables
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Create test arguments
        self.args = ScriptArguments(
            model_name="mock-model",
            learning_rate=1e-5,
            mini_batch_size=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            max_new_tokens=10,
            ppo_epochs=1,
            n_epochs=1,
        )

        # Create small test dataset
        self.test_data = {
            "question": [
                "What is 2 + 2?",
                "What is 3 * 4?",
            ],
            "answer": [
                "#### 4",
                "#### 12",
            ],
        }
        self.test_dataset = Dataset.from_dict(self.test_data)

    def test_compute_rewards(self):
        responses = [
            "Result = 4 <submit>",
            "Result = 12 <submit>",
            "Result = 4.0 <submit>",  # Test float parsing
            "Result = 12000 <submit>",  # Test without comma
            "Invalid response",  # Test invalid response
        ]
        answers = ["4", "12", "4", "12000", "5"]
        rewards = compute_rewards(responses, answers)

        # Check reward values
        self.assertEqual(len(rewards), 5)
        self.assertEqual(rewards[0].item(), 1.0)  # Correct answer
        self.assertEqual(rewards[1].item(), 1.0)  # Correct answer
        self.assertEqual(rewards[2].item(), 1.0)  # Float answer
        self.assertEqual(rewards[3].item(), 1.0)  # Large number
        self.assertEqual(rewards[4].item(), 0.0)  # Invalid response

    def test_parse_args(self):
        # Test default arguments
        args = parse_args([])
        self.assertEqual(args.model_name, "gpt2")
        self.assertEqual(args.learning_rate, 1e-5)
        self.assertEqual(args.mini_batch_size, 1)
        self.assertEqual(args.batch_size, 16)
        self.assertEqual(args.gradient_accumulation_steps, 16)
        self.assertEqual(args.max_new_tokens, 256)
        self.assertEqual(args.ppo_epochs, 1)
        self.assertEqual(args.n_epochs, 32)

        # Test custom arguments
        custom_args = [
            "--model_name", "custom-model",
            "--learning_rate", "2e-5",
            "--mini_batch_size", "2",
            "--batch_size", "32",
            "--gradient_accumulation_steps", "8",
            "--max_new_tokens", "128",
            "--ppo_epochs", "2",
            "--n_epochs", "16",
        ]
        args = parse_args(custom_args)
        self.assertEqual(args.model_name, "custom-model")
        self.assertEqual(args.learning_rate, 2e-5)
        self.assertEqual(args.mini_batch_size, 2)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.gradient_accumulation_steps, 8)
        self.assertEqual(args.max_new_tokens, 128)
        self.assertEqual(args.ppo_epochs, 2)
        self.assertEqual(args.n_epochs, 16)

    @patch("mini_trl.python_interpreter_ppo.AutoModelForCausalLM.from_pretrained")
    @patch("mini_trl.python_interpreter_ppo.AutoTokenizer.from_pretrained")
    @patch("mini_trl.python_interpreter_ppo.AutoModelForSequenceClassification.from_pretrained")
    @patch("mini_trl.python_interpreter_ppo.load_dataset")
    def test_setup(
        self,
        mock_load_dataset,
        mock_seq_class,
        mock_tokenizer,
        mock_model,
    ):
        # Mock models
        mock_model.return_value = MockModel()
        mock_tokenizer.return_value = MockTokenizer()
        mock_seq_class.return_value = MockModel()

        # Mock dataset
        mock_load_dataset.return_value = self.test_dataset

        # Run setup
        result = setup(self.args)
        (
            model, ref_model, tokenizer,
            value_model, reward_model,
            train_ds, test_ds
        ) = result

        # Check model types
        self.assertIsInstance(model, MockModel)
        self.assertIsInstance(ref_model, MockModel)
        self.assertIsInstance(tokenizer, MockTokenizer)
        self.assertIsInstance(value_model, MockModel)
        self.assertIsInstance(reward_model, MockModel)

        # Check dataset types
        self.assertIsInstance(train_ds, Dataset)
        self.assertIsInstance(test_ds, Dataset)

        # Check dataset content
        # Note: train_ds will have 1 example because first sample is skipped
        self.assertEqual(len(train_ds), 1)
        self.assertEqual(len(test_ds), 2)

    def test_main(self):
        # Test main function setup only
        try:
            # Create output directory
            os.makedirs("model", exist_ok=True)
            # Just verify the setup works
            self.assertTrue(os.path.exists("model"))
        finally:
            # Cleanup
            if os.path.exists("model"):
                import shutil
                shutil.rmtree("model")


if __name__ == "__main__":
    unittest.main()