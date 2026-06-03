# -*- coding: utf-8 -*-
"""NSMC 감성 분류 미세 조정 과제 템플릿."""

from pathlib import Path
import csv
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    from .model import GPTModel
except ImportError:
    from model import GPTModel


def make_sentiment_dataset(
    train_tsv_path: str | Path,
    test_tsv_path: str | Path | None = None,
    val_ratio: float = 0.08,
    seed: int = 42,
    output_dir: str | Path | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    TODO: NSMC TSV를 읽어 train/validation/test 감성 분류 데이터를 만듭니다.

    반환 형식:
        [{"text": "리뷰", "label": 0 또는 1}, ...]
    """
    def read_nsmc_tsv(path: str | Path) -> list[dict]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                text = (row.get("document") or "").strip()
                label = row.get("label")
                if not text or label is None:
                    continue
                try:
                    label_id = int(label)
                except ValueError:
                    continue
                rows.append({"text": text, "label": label_id})
        return rows

    train_val_data = read_nsmc_tsv(train_tsv_path)
    rng = random.Random(seed)
    rng.shuffle(train_val_data)

    val_size = int(len(train_val_data) * val_ratio)
    val_data = train_val_data[:val_size]
    train_data = train_val_data[val_size:]
    test_data = read_nsmc_tsv(test_tsv_path) if test_tsv_path is not None else []

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        for name, dataset in (
            ("train", train_data),
            ("val", val_data),
            ("test", test_data),
        ):
            with open(output_path / f"nsmc_sentiment_{name}.jsonl", "w", encoding="utf-8") as f:
                for item in dataset:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return train_data, val_data, test_data


class ReviewSentimentDataset(Dataset):
    """감성 분류용 Dataset. 리뷰 하나와 label 하나를 반환합니다."""

    def __init__(
        self,
        data: list[dict],
        tokenizer,
        max_length: int = 128,
        pad_id: int | None = None,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.get_pad_id() if pad_id is None else pad_id

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """TODO: text를 encode하고 max_length까지 자르거나 padding한 뒤 label과 함께 반환합니다."""
        item = self.data[idx]
        token_ids = self.tokenizer.encode(item["text"])
        token_ids = token_ids[:self.max_length]
        token_ids = token_ids + [self.pad_id] * (self.max_length - len(token_ids))
        return torch.tensor(token_ids, dtype=torch.long), int(item["label"])


class GPTForSequenceClassification(nn.Module):
    """
    GPT backbone 위에 감성 분류용 Linear head를 붙인 모델.

    주의: LM head는 다음 토큰 예측용입니다. 감성 분류는 hidden state 위에 별도 classifier를 붙입니다.
    """

    def __init__(
        self,
        gpt_model: GPTModel,
        num_labels: int = 2,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.gpt = gpt_model
        self.num_labels = num_labels
        # TODO: dropout과 classifier를 정의하세요. classifier 입력 차원은 gpt_model.config["emb_dim"]입니다.
        self.dropout = nn.Dropout(drop_rate)
        self.classifier = nn.Linear(gpt_model.config["emb_dim"], num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: GPT hidden state에서 문장 대표 벡터를 뽑아 분류 logits를 만듭니다.

        labels가 있으면 (loss, logits), 없으면 logits를 반환합니다.
        """
        batch_size, seq_len = input_ids.shape
        pos_ids = torch.arange(seq_len, device=input_ids.device)

        x = self.gpt.tok_emb(input_ids) + self.gpt.pos_emb(pos_ids)
        x = self.gpt.drop_emb(x)
        x = self.gpt.trf_blocks(x)
        x = self.gpt.final_norm(x)

        pooled = x[:, -1, :]
        logits = self.classifier(self.dropout(pooled))

        if labels is None:
            return logits

        labels = labels.to(logits.device).long()
        loss = nn.functional.cross_entropy(logits, labels)
        return loss, logits


def train_epoch_sentiment(
    model: GPTForSequenceClassification,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 1 epoch 훈련하고 (평균 loss, accuracy)를 반환합니다."""
    model.train()
    total_loss = 0.0
    total_examples = 0
    correct = 0

    for input_batch, target_batch in train_loader:
        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device).long()

        optimizer.zero_grad()
        loss, logits = model(input_batch, target_batch)
        loss.backward()
        optimizer.step()

        batch_size = target_batch.shape[0]
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        correct += (logits.argmax(dim=-1) == target_batch).sum().item()

    if total_examples == 0:
        return float("nan"), float("nan")
    return total_loss / total_examples, correct / total_examples


def evaluate_sentiment(
    model: GPTForSequenceClassification,
    data_loader,
    device: torch.device,
) -> tuple[float, float]:
    """TODO: 감성 분류 모델을 평가하고 (평균 loss, accuracy)를 반환합니다."""
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_examples = 0
    correct = 0

    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device).long()

            loss, logits = model(input_batch, target_batch)
            batch_size = target_batch.shape[0]
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            correct += (logits.argmax(dim=-1) == target_batch).sum().item()

    if was_training:
        model.train()

    if total_examples == 0:
        return float("nan"), float("nan")
    return total_loss / total_examples, correct / total_examples
