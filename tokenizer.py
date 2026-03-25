import os
import re
import glob
import random
import argparse
import tempfile
from typing import List

VOCAB_SIZE = 5000
MODEL_PREFIX = "hindi_bpe"  # saves hindi_bpe.model + hindi_bpe.vocab
MODEL_PATH = f"{MODEL_PREFIX}.model"
CORPUS_DIR = "hindi_corpus/train"
TRAIN_RATIO = 0.9
RANDOM_SEED = 42

# Special token IDs assigned by SentencePiece trainer:
#   0 = <unk> / PAD,  1 = <s> (BOS),  2 = </s> (EOS)
# pad_id is disabled in the trainer, we reuse UNK as PAD.
UNK_ID = 0
BOS_ID = 1
EOS_ID = 2
PAD_ID = 0  # same as UNK, mask pads in attention when building the model


def clean_text(text: str) -> str:
    # Retain only Devanagari + digits + whitespace
    text = re.sub(r'[^\u0900-\u097F0-9\s]', ' ', text)
    # Collapse runs of whitespace
    text = re.sub(r'[ \t\r]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def get_split_files(corpus_dir: str = CORPUS_DIR,
                    train_ratio: float = TRAIN_RATIO,
                    seed: int = RANDOM_SEED):
    """Return (train_files, val_files) via a deterministic shuffle-split."""
    all_files = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))
    random.seed(seed)
    random.shuffle(all_files)
    n_train = int(len(all_files) * train_ratio)
    return all_files[:n_train], all_files[n_train:]


def read_files(file_list: List[str]) -> List[str]:
    """Read and clean texts from a list of file paths; skip empty/broken ones."""
    texts = []
    for path in file_list:
        try:
            with open(path, encoding='utf-8', errors='ignore') as f:
                text = clean_text(f.read())
            if text:
                texts.append(text)
        except Exception:
            pass
    return texts


def train_tokenizer(corpus_dir: str = CORPUS_DIR,
                    vocab_size: int = VOCAB_SIZE,
                    model_prefix: str = MODEL_PREFIX):
    """Train a SentencePiece BPE model and save .model / .vocab to disk."""
    import sentencepiece as spm

    train_files, _ = get_split_files(corpus_dir)
    print(f"Training tokenizer on {len(train_files)} files …")

    # dump all training text into a temp file for SentencePiece
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt',
                                     encoding='utf-8', delete=False) as tmp:
        tmp_path = tmp.name
        for path in train_files:
            try:
                with open(path, encoding='utf-8', errors='ignore') as f:
                    text = clean_text(f.read())
                if text:
                    tmp.write(text + '\n')
            except Exception:
                pass

    try:
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=model_prefix,
            model_type='bpe',
            vocab_size=vocab_size,
            character_coverage=0.9995,  # high coverage keeps rare Hindi chars
            pad_id=-1,                  # padding handled at the dataset level
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            unk_id=UNK_ID,
            bos_piece='<s>',
            eos_piece='</s>',
            unk_piece='<unk>',
        )
    finally:
        os.remove(tmp_path)

    print(f"Tokenizer saved to {model_prefix}.model  (vocab_size={vocab_size})")


# Module-level singleton so we only load the model once
_tokenizer = None


def get_tokenizer(model_path: str = MODEL_PATH):
    """Load and cache the SentencePiece model. Raises if not yet trained."""
    global _tokenizer
    if _tokenizer is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Tokenizer model not found at '{model_path}'.\n"
                f"Run: python tokenizer.py --train"
            )
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        _tokenizer = sp
    return _tokenizer


def encode(text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
    """Tokenise text and return a list of integer token IDs."""
    sp = get_tokenizer()
    ids = sp.encode(text, out_type=int)
    if add_bos:
        ids = [BOS_ID] + ids
    if add_eos:
        ids = ids + [EOS_ID]
    return ids


def decode(ids: List[int]) -> str:
    """Convert token IDs back to a string, stripping special tokens."""
    sp = get_tokenizer()
    filtered = [i for i in ids if i not in (PAD_ID, BOS_ID, EOS_ID)]
    return sp.decode(filtered)


def vocab_size() -> int:
    """Return the vocabulary size of the loaded tokenizer."""
    return get_tokenizer().get_piece_size()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hindi BPE Tokenizer")
    parser.add_argument('--train', action='store_true',
                        help='Train the tokenizer on the corpus')
    parser.add_argument('--corpus_dir', default=CORPUS_DIR)
    parser.add_argument('--vocab_size', type=int, default=VOCAB_SIZE)
    parser.add_argument('--model_prefix', default=MODEL_PREFIX)
    args = parser.parse_args()

    if args.train:
        train_tokenizer(args.corpus_dir, args.vocab_size, args.model_prefix)
    
    # demo
    tok = get_tokenizer()
    sample = "मेरा नाम आर्य है और मुझे फिल्में देखना पसंद है।"
    ids = encode(sample)
    decoded = decode(ids)
    print(f"Sample : {sample}")
    print(f"IDs    : {ids[:10]} … (len={len(ids)})")
    print(f"Decoded: {decoded}")
    print(f"Vocab  : {vocab_size()} tokens")
