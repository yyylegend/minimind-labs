import math
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from minimind_lab.config import MiniMindConfig
from trainer.trainer_utils import EpochRandomSampler, _build_resumed_dataloader, train_model


class IndexedDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size
        self.accessed: list[int] = []

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> torch.Tensor:
        self.accessed.append(index)
        return torch.tensor(index)


class TinyOutput:
    def __init__(self, loss: torch.Tensor, aux_loss: torch.Tensor) -> None:
        self.loss = loss
        self.aux_loss = aux_loss


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 8, hidden_size: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits = self.lm_head(self.embedding(input_ids))
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return TinyOutput(loss, logits.sum() * 0.0)


class ResumeDataLoaderTest(unittest.TestCase):
    def test_resume_loader_does_not_read_completed_batches(self):
        dataset = IndexedDataset(size=11)
        sampler = EpochRandomSampler(dataset, seed=7)
        sampler.set_epoch(3)
        order = list(iter(sampler))
        dataloader = DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=0)

        resumed = _build_resumed_dataloader(dataloader, skip_batches=2)
        resumed_items = [int(item) for batch in resumed for item in batch]

        expected = order[4:]
        self.assertEqual(resumed_items, expected)
        self.assertEqual(dataset.accessed, expected)
        self.assertEqual(len(resumed), math.ceil(len(expected) / 2))

    def test_train_model_resume_processes_only_remaining_batches(self):
        config = MiniMindConfig(
            vocab_size=8,
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            max_position_embeddings=4,
        )

        class TrainingDataset(IndexedDataset):
            def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                self.accessed.append(index)
                ids = torch.tensor([(index + offset) % 8 for offset in range(4)])
                return {
                    "input_ids": ids,
                    "labels": ids.clone(),
                    "attention_mask": torch.ones_like(ids),
                }

        def make_loader(dataset):
            sampler = EpochRandomSampler(dataset, seed=11)
            return DataLoader(dataset, batch_size=2, sampler=sampler, num_workers=0)

        with tempfile.TemporaryDirectory() as output_dir:
            first_dataset = TrainingDataset(size=5)
            train_model(
                model=TinyLanguageModel(),
                dataloader=make_loader(first_dataset),
                config=config,
                device=torch.device("cpu"),
                dtype=torch.float32,
                output_dir=output_dir,
                stage="pretrain",
                epochs=1,
                learning_rate=1e-3,
                accumulation_steps=1,
                grad_clip=1.0,
                save_interval=0,
                max_steps=2,
                log_interval=100,
            )

            resumed_dataset = TrainingDataset(size=5)
            train_model(
                model=TinyLanguageModel(),
                dataloader=make_loader(resumed_dataset),
                config=config,
                device=torch.device("cpu"),
                dtype=torch.float32,
                output_dir=output_dir,
                stage="pretrain",
                epochs=1,
                learning_rate=1e-3,
                accumulation_steps=1,
                grad_clip=1.0,
                save_interval=0,
                max_steps=0,
                log_interval=100,
                resume_checkpoint=f"{output_dir}/pretrain_last.pt",
            )

            sampler = EpochRandomSampler(TrainingDataset(size=5), seed=11)
            expected_order = list(iter(sampler))
            self.assertEqual(resumed_dataset.accessed, expected_order[4:])


if __name__ == "__main__":
    unittest.main()
