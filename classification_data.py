import os
import csv
import random
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset

from tokenizer import encode, PAD_ID, clean_text

CSV_PATH = "text_classification_dataset/train.csv"
MAX_LENGTH = 256
TRAIN_RATIO = 0.9
RANDOM_SEED = 42

TEXT_COL = 'text'
LABEL_COL = 'experience'  # 0 = Negative, 1 = Neutral, 2 = Positive


def _load_csv(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, encoding='utf-8', errors='ignore') as f:
        rows = list(csv.DictReader(f))
    return rows


def _stratified_split(rows: List[Dict],
                      train_ratio: float = TRAIN_RATIO,
                      seed: int = RANDOM_SEED) -> Tuple[List[Dict], List[Dict]]:
    """Shuffle-split rows while preserving class proportions."""
    buckets: Dict[str, List] = defaultdict(list)
    for row in rows:
        buckets[row[LABEL_COL]].append(row)

    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for label, bucket in buckets.items():
        rng.shuffle(bucket)
        n_train = max(1, int(len(bucket) * train_ratio))
        train_rows.extend(bucket[:n_train])
        val_rows.extend(bucket[n_train:])

    # Re-shuffle so samples aren't class-sorted
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def _pad_or_truncate(ids: List[int], max_length: int) -> List[int]:
    ids = ids[:max_length]
    ids += [PAD_ID] * (max_length - len(ids))
    return ids


class ClassificationDataset(Dataset):
    """
    Sentiment classification dataset for Hindi movie reviews.

    Returns (x, y): x is a LongTensor of shape (max_length,), y is an int in {0, 1, 2}.
    """

    def __init__(self,
                 split: str = 'train',
                 max_length: int = MAX_LENGTH,
                 csv_path: str = CSV_PATH):
        assert split in ('train', 'val'), "split must be 'train' or 'val'"

        self.max_length = max_length

        all_rows = _load_csv(csv_path)
        train_rows, val_rows = _stratified_split(all_rows)
        rows = train_rows if split == 'train' else val_rows
        print(f"[ClassificationDataset] {split}: {len(rows)} samples")

        label_counts = Counter(r[LABEL_COL] for r in rows)
        print(f"[ClassificationDataset] {split} label distribution: {dict(label_counts)}")

        self._samples: List[Tuple[torch.Tensor, int]] = []
        for row in rows:
            text = clean_text(row.get(TEXT_COL, ''))
            label = int(row[LABEL_COL])
            ids = encode(text)
            ids = _pad_or_truncate(ids, max_length)
            x = torch.tensor(ids, dtype=torch.long)
            self._samples.append((x, label))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self._samples[idx]


def get_classification_loaders(max_length: int = MAX_LENGTH,
                                batch_size: int = 32,
                                num_workers: int = 0,
                                csv_path: str = CSV_PATH):
    """Build and return (train_loader, val_loader)."""
    from torch.utils.data import DataLoader

    train_ds = ClassificationDataset('train', max_length, csv_path)
    val_ds = ClassificationDataset('val', max_length, csv_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


if __name__ == '__main__':
    train_ds = ClassificationDataset(split='train', max_length=128)
    val_ds = ClassificationDataset(split='val', max_length=128)
    x, y = train_ds[0]
    print(f"x shape : {x.shape}")
    print(f"label   : {y}  (type={type(y).__name__})")
    assert x.shape == (128,), "x must be shape (max_length,)"
    assert y in (0, 1, 2), "label must be in {0, 1, 2}"
    assert x.dtype == torch.long
    print("All checks passed ✓")
