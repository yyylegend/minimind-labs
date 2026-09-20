import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from minimind_lab.config import MiniMindConfig
from trainer.trainer_utils import cosine_learning_rate, save_checkpoint, train_model


class OneBatchDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor([1, 2, 3, 4])
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


class NaNLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, input_ids, attention_mask=None, labels=None):
        loss = self.weight * torch.tensor(float("nan"))
        return SimpleNamespace(loss=loss, aux_loss=torch.zeros_like(loss))


class TrainingStabilityTest(unittest.TestCase):
    def test_cosine_schedule_warms_up_and_reaches_floor(self):
        base_lr = 3e-4
        warmup_steps = 100
        total_steps = 1000

        first_lr = cosine_learning_rate(1, total_steps, base_lr, warmup_steps, 0.1)
        warmup_end_lr = cosine_learning_rate(100, total_steps, base_lr, warmup_steps, 0.1)
        middle_lr = cosine_learning_rate(500, total_steps, base_lr, warmup_steps, 0.1)
        final_lr = cosine_learning_rate(1000, total_steps, base_lr, warmup_steps, 0.1)

        self.assertLess(first_lr, warmup_end_lr)
        self.assertAlmostEqual(warmup_end_lr, base_lr)
        self.assertGreater(middle_lr, final_lr)
        self.assertAlmostEqual(final_lr, base_lr * 0.1)

    def test_save_checkpoint_refuses_nonfinite_model(self):
        model = nn.Linear(2, 2)
        model.weight.data.fill_(float("nan"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        config = MiniMindConfig(
            vocab_size=8,
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            max_position_embeddings=4,
        )

        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaises(FloatingPointError):
                save_checkpoint(
                    model,
                    optimizer,
                    scaler,
                    config,
                    output_dir,
                    "pretrain",
                    epoch=0,
                    step=1,
                    batch_index=1,
                )
            self.assertFalse(Path(output_dir, "pretrain_last.pt").exists())

    def test_train_model_stops_before_backward_on_nonfinite_loss(self):
        config = MiniMindConfig(
            vocab_size=8,
            hidden_size=4,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            max_position_embeddings=4,
        )
        dataloader = DataLoader(OneBatchDataset(), batch_size=1, num_workers=0)

        with tempfile.TemporaryDirectory() as output_dir:
            with self.assertRaises(FloatingPointError):
                train_model(
                    model=NaNLossModel(),
                    dataloader=dataloader,
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
                    max_steps=1,
                    log_interval=1,
                )
            self.assertFalse(Path(output_dir, "pretrain_last.pt").exists())


if __name__ == "__main__":
    unittest.main()
