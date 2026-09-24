import json
import tempfile
import unittest
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import torch

from minimind_lab import MiniMindConfig, MiniMindForCausalLM
from trainer.grpo_rollout import TorchRolloutEngine
from trainer.train_grpo import _pilot, _train


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 19

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None, return_tensors=None):
        ids = [1 + (ord(character) % 17) for character in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=True):
        tokens = [int(token) for token in token_ids]
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in {self.pad_token_id, self.eos_token_id}]
        return " ".join(str(token) for token in tokens)


class TwoPromptDataset:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        prompt = f"question {index}"
        return {"prompt": prompt, "messages": [{"role": "user", "content": prompt}]}


class AlternatingRewardModel:
    def __init__(self):
        self.calls = 0

    def get_score(self, messages, response):
        score = 0.0 if self.calls % 2 == 0 else 2.0
        self.calls += 1
        return score


class GRPOTrainingSmokeTest(unittest.TestCase):
    def _make_model_and_engine(self):
        tokenizer = TinyTokenizer()
        config = MiniMindConfig(
            vocab_size=20,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=12,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = MiniMindForCausalLM(config)
        engine = TorchRolloutEngine(
            model,
            tokenizer,
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_seq_len=8,
            max_gen_len=3,
            temperature=1.0,
        )
        return config, model, engine

    def test_rollout_only_writes_grouped_samples_and_pilot_summary(self):
        torch.manual_seed(13)
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                pilot_prompts=2,
                batch_size=2,
                num_generations=2,
                pilot_output_dir=directory,
            )
            _, _, engine = self._make_model_and_engine()

            _pilot(
                args,
                TwoPromptDataset(),
                engine,
                AlternatingRewardModel(),
                torch.device("cpu"),
            )

            summary = json.loads(Path(directory, "summary.json").read_text(encoding="utf-8"))
            samples = Path(directory, "rollouts.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(summary["prompts"], 2)
            self.assertEqual(summary["completions"], 4)
            self.assertEqual(summary["zero_variance_groups"], 0)
            self.assertEqual(len(samples), 2)

    def test_one_optimizer_step_updates_policy_and_saves_resumable_checkpoint(self):
        torch.manual_seed(11)
        config, policy, rollout_engine = self._make_model_and_engine()
        reference = deepcopy(policy).eval()
        reference.requires_grad_(False)
        before = [parameter.detach().clone() for parameter in policy.parameters()]
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(
                batch_size=2,
                num_generations=2,
                num_workers=0,
                seed=42,
                learning_rate=1e-3,
                tensorboard_dir="",
                resume_checkpoint="",
                epochs=1,
                max_steps=1,
                save_interval=1,
                log_interval=1,
                grad_clip=1.0,
                min_lr_ratio=0.1,
                beta=0.1,
                epsilon_high=5.0,
                output_dir=directory,
            )

            _train(
                args,
                TwoPromptDataset(),
                policy,
                reference,
                rollout_engine,
                AlternatingRewardModel(),
                config,
                torch.device("cpu"),
                torch.float32,
            )

            checkpoint_path = Path(directory, "grpo_last.pt")
            self.assertTrue(checkpoint_path.is_file())
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["step"], 1)
            self.assertTrue(
                any(not torch.equal(old, new) for old, new in zip(before, policy.parameters()))
            )


if __name__ == "__main__":
    unittest.main()
