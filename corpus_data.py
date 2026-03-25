import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from tqdm import tqdm

from tokenizer import get_split_files, read_files, encode, BOS_ID

DEFAULT_CONTEXT = 256
DEFAULT_STRIDE = 1  # stride=1 maximum overlap, increase to save RAM
CORPUS_DIR = "hindi_corpus/train"


class HindiCorpusDataset(Dataset):
    """
    Language modelling dataset over the Hindi corpus.
    Each item is (x, y) — both LongTensors of shape (context_length,).
    y is x shifted right by one token (next-token prediction).
    """

    def __init__(self,
                 split: str = 'train',
                 context_length: int = DEFAULT_CONTEXT,
                 stride: int = DEFAULT_STRIDE,
                 corpus_dir: str = CORPUS_DIR):
        assert split in ('train', 'val'), "split must be 'train' or 'val'"

        self.context_length = context_length
        self.stride = stride

        # Get the file list for this split
        train_files, val_files = get_split_files(corpus_dir)
        file_list = train_files if split == 'train' else val_files
        print(f"[HindiCorpusDataset] {split}: {len(file_list):,} files")

        # check for cached dataset to bypass tokenization overhead
        cache_path = os.path.join(corpus_dir, f"{split}_cache.pt")
        if os.path.exists(cache_path):
            print(f"[HindiCorpusDataset] Loading cached {split} data from {cache_path}")
            self._data = torch.load(cache_path)
            print(f"[HindiCorpusDataset] {split}: {len(self._data):,} tokens, "
                  f"{len(self):,} samples (stride={stride})")
            return

        # tokenize, prepend BOS per document
        print(f"[HindiCorpusDataset] Tokenising {split} files …")
        all_ids = []
        for path in tqdm(file_list, desc=f"Tokenising {split}"):
            try:
                with open(path, encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
            except Exception:
                continue
            if not text:
                continue
            ids = [BOS_ID] + encode(text)
            all_ids.extend(ids)

        if len(all_ids) < context_length + 1:
            raise ValueError(
                f"Not enough tokens ({len(all_ids)}) for context_length={context_length}. "
                "Try a smaller context_length or use more data."
            )

        # store as a flat tensor for fast slicing
        self._data = torch.tensor(all_ids, dtype=torch.long)
        print(f"[HindiCorpusDataset] Saving tokenized cache to {cache_path}")
        torch.save(self._data, cache_path)
        print(f"[HindiCorpusDataset] {split}: {len(self._data):,} tokens, "
              f"{len(self):,} samples (stride={stride})")

    def __len__(self) -> int:
        return (len(self._data) - self.context_length) // self.stride

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        x = self._data[start: start + self.context_length]
        y = self._data[start + 1: start + self.context_length + 1]
        return x, y


def get_corpus_loaders(context_length: int = DEFAULT_CONTEXT,
                       stride: int = DEFAULT_STRIDE,
                       batch_size: int = 32,
                       num_workers: int = 0,
                       corpus_dir: str = CORPUS_DIR):
    """Build and return (train_loader, val_loader)."""
    

    train_ds = HindiCorpusDataset('train', context_length, stride, corpus_dir)
    val_ds = HindiCorpusDataset('val', context_length, stride, corpus_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


if __name__ == '__main__':
    ds = HindiCorpusDataset(split='train', context_length=64, stride=64)
    x, y = ds[0]
    print(f"x shape : {x.shape}")
    print(f"y shape : {y.shape}")
    print(f"x[:5]   : {x[:5].tolist()}")
    print(f"y[:5]   : {y[:5].tolist()}")
    assert (x[1:] == y[:-1]).all(), "y must be x shifted by 1!"
    print("Shift check passed.")
