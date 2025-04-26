import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import torch
from datasets import Dataset

from python_interpreter_ppo import (
    ScriptArguments,
    exact_match_reward,
    setup,
    main,
)
from .trl_shim import PPOConfig, PPOTrainer


class TestPythonInterpreterPPO(unittest.TestCase):
    def setUp(self):
        # Mock environment variables
        self.env_patcher = patch.dict(
            os.environ,
            {
                "HF_ALLOW_CODE_EVAL": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
        )
        self.env_patcher.start()

        # Create mock arguments
        self.args = ScriptArguments(
            model_name="test-model",
            learning_rate=1e-5,
            mini_batch_size=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            max_new_tokens=10,
            ppo_epochs=1,
            n_epochs=1,
        )

    def tearDown(self):
        self.env_patcher.stop()

    def test_exact_match_reward(self):
        responses = [
            "Result = 42 <submit>",
            "Result = 43 <submit>",
            "Invalid response",
        ]
        answers = ["42", "42", "42"]
        rewards = exact_match_reward(responses, answers)
        self.assertEqual(len(rewards), 3)
        self.assertEqual(rewards[0].item(), 1.0)  # Correct answer
        self.assertEqual(rewards[1].item(), 0.0)  # Wrong answer
        self.assertEqual(rewards[2].item(), 0.0)  # Invalid response

    @patch("python_interpreter_ppo.AutoModelForCausalLM.from_pretrained")
    @patch("python_interpreter_ppo.AutoTokenizer.from_pretrained")
    @patch("python_interpreter_ppo.AutoModelForSequenceClassification.from_pretrained")
    @patch("python_interpreter_ppo.load_dataset")
    def test_setup(
        self,
        mock_load_dataset,
        mock_seq_class,
        mock_tokenizer,
        mock_model,
    ):
        # Mock dataset
        mock_ds = Dataset.from_dict(
            {
                "question": ["q1", "q2"],
                "answer": ["#### 42", "#### 43"],
            }
        )
        mock_load_dataset.return_value = mock_ds

        # Mock models
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_seq_class.return_value = MagicMock()

        # Run setup
        result = setup(self.args)

        # Check return values
        self.assertEqual(len(result), 7)
        self.assertIsInstance(result[0], MagicMock)  # model
        self.assertIsInstance(result[1], MagicMock)  # ref_model
        self.assertIsInstance(result[2], MagicMock)  # tokenizer
        self.assertIsInstance(result[3], MagicMock)  # value_model
        self.assertIsInstance(result[4], MagicMock)  # reward_model
        self.assertIsInstance(result[5], Dataset)  # train dataset
        self.assertIsInstance(
            result[6], torch.utils.data.DataLoader
        )  # test dataloader

    @patch("python_interpreter_ppo.setup")
    @patch("python_interpreter_ppo.PPOTrainer")
    @patch("python_interpreter_ppo.TextEnvironment")
    @patch("python_interpreter_ppo.load_tool")
    def test_main(
        self,
        mock_load_tool,
        mock_text_env,
        mock_ppo_trainer,
        mock_setup,
    ):
        # Mock setup return values
        mock_model = MagicMock()
        mock_ref_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_value_model = MagicMock()
        mock_reward_model = MagicMock()
        mock_ds = Dataset.from_dict({"query": ["q1"], "answer": ["42"]})
        mock_dataloader = MagicMock()
        mock_setup.return_value = (
            mock_model,
            mock_ref_model,
            mock_tokenizer,
            mock_value_model,
            mock_reward_model,
            mock_ds,
            mock_dataloader,
        )

        # Mock PPO trainer
        mock_trainer = MagicMock()
        mock_trainer.dataloader = [
            {"query": ["q1"], "answer": ["42"]},
        ]
        mock_trainer.accelerator = MagicMock()
        mock_trainer.accelerator.prepare.return_value = mock_dataloader
        mock_ppo_trainer.return_value = mock_trainer

        # Mock text environment
        mock_env = MagicMock()
        mock_env.run.return_value = (
            ["q1"],
            ["response"],
            [True],
            [torch.tensor(1.0)],
            [{}],
        )
        mock_text_env.return_value = mock_env

        # Mock load_tool
        mock_tool = MagicMock()
        mock_load_tool.return_value = mock_tool

        # Run main
        with tempfile.TemporaryDirectory():
            main(self.args)

        # Verify trainer was created
        mock_ppo_trainer.assert_called_once()
        # Verify training loop ran
        self.assertTrue(mock_trainer.step.called)
        # Verify final evaluation
        self.assertTrue(mock_env.run.called)
        # Verify tool was loaded
        mock_load_tool.assert_called_once_with("lvwerra/python-interpreter")


if __name__ == "__main__":
    unittest.main()