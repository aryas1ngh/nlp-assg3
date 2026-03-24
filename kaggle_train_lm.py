import os
import re
import glob
import random
import tempfile
import json
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ==========================================
# 1. Config & Constants
# ==========================================
VOCAB_SIZE = 5000
MODEL_PREFIX = "hindi_bpe"
MODEL_PATH = f"{MODEL_PREFIX}.model"
CORPUS_DIR = "hindi_corpus/train"
TRAIN_RATIO = 0.9
RANDOM_SEED = 42

UNK_ID = 0
BOS_ID = 1
EOS_ID = 2
PAD_ID = 0

OUTPUT_FILE = 'generated_text.txt'

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ==========================================
# 2. Tokenizer (SentencePiece)
# ==========================================
def clean_text(text: str) -> str:
    """Keep only Devanagari script + Hindi punctuation + whitespace."""
    text = re.sub(r'[^\u0900-\u097F0-9\s]', ' ', text)
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

def train_tokenizer(corpus_dir: str = CORPUS_DIR,
                    vocab_size: int = VOCAB_SIZE,
                    model_prefix: str = MODEL_PREFIX):
    """Train a SentencePiece BPE model and save .model / .vocab to disk."""
    import sentencepiece as spm

    train_files, _ = get_split_files(corpus_dir)
    print(f"Training tokenizer on {len(train_files)} files …")

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
            character_coverage=0.9995,
            pad_id=-1,
            bos_id=BOS_ID,
            eos_id=EOS_ID,
            unk_id=UNK_ID,
            bos_piece='<s>',
            eos_piece='</s>',
            unk_piece='<unk>',
        )
    finally:
        os.remove(tmp_path)

    print(f"Tokenizer saved → {model_prefix}.model  (vocab_size={vocab_size})")

_tokenizer = None

def get_tokenizer(model_path: str = MODEL_PATH):
    """Load and cache the SentencePiece model."""
    global _tokenizer
    if _tokenizer is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Tokenizer model not found at '{model_path}'.\n"
                f"Run tokenizer training first."
            )
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        _tokenizer = sp
    return _tokenizer

def encode(text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
    """Tokenize text and return a list of integer token IDs."""
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
    return get_tokenizer().get_piece_size()


# ==========================================
# 3. Dataset & DataLoader
# ==========================================
class HindiCorpusDataset(Dataset):
    def __init__(self,
                 split: str = 'train',
                 context_length: int = 256,
                 stride: int = 1,
                 corpus_dir: str = CORPUS_DIR):
        assert split in ('train', 'val'), "split must be 'train' or 'val'"

        self.context_length = context_length
        self.stride = stride

        train_files, val_files = get_split_files(corpus_dir)
        file_list = train_files if split == 'train' else val_files
        print(f"[HindiCorpusDataset] {split}: {len(file_list):,} files")

        cache_path = os.path.join(corpus_dir, f"{split}_cache.pt")
        if os.path.exists(cache_path):
            print(f"[HindiCorpusDataset] Loading cached {split} data from {cache_path}")
            self._data = torch.load(cache_path)
            print(f"[HindiCorpusDataset] {split}: {len(self._data):,} tokens, "
                  f"{len(self):,} samples (stride={stride})")
            return

        print(f"[HindiCorpusDataset] Tokenizing {split} files …")
        all_ids = []
        for path in tqdm(file_list, desc=f"Tokenizing {split}"):
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

def get_corpus_loaders(context_length: int = 256,
                       stride: int = 1,
                       batch_size: int = 32,
                       num_workers: int = 0,
                       corpus_dir: str = CORPUS_DIR):
    train_ds = HindiCorpusDataset('train', context_length, stride, corpus_dir)
    val_ds = HindiCorpusDataset('val', context_length, stride, corpus_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


# ==========================================
# 4. Model Architecture (MiniGPT)
# ==========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        def split_heads(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(self.q_proj(x))
        K = split_heads(self.k_proj(x))
        V = split_heads(self.v_proj(x))

        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_sz: int, context_length: int, embed_dim: int,
                 num_heads: int, num_layers: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.context_length = context_length
        self.token_emb = nn.Embedding(vocab_sz, embed_dim)
        self.pos_emb = nn.Embedding(context_length, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.context_length, f"Sequence length {T} exceeds context_length {self.context_length}"
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(tok + pos)
        x = self.blocks(x)
        return self.ln_final(x)

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_sz: int = None, context_length: int = 256,
                 embed_dim: int = 512, num_heads: int = 8,
                 num_layers: int = 8, ffn_dim: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        if vocab_sz is None:
            vocab_sz = vocab_size()

        self.backbone = MiniGPT(vocab_sz, context_length, embed_dim, num_heads, num_layers, ffn_dim, dropout)
        self.lm_head = nn.Linear(embed_dim, vocab_sz, bias=False)
        self.lm_head.weight = self.backbone.token_emb.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        hidden = self.backbone(idx)
        logits = self.lm_head(hidden)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 50,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        self.eval()
        ctx_len = self.backbone.context_length
        idx = prompt_ids.clone()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -ctx_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx

# ==========================================
# 5. Training Utils
# ==========================================
def get_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(model: GPTLanguageModel, train_loader: DataLoader, val_loader: DataLoader,
          num_epochs: int, lr: float):
    total_steps   = num_epochs * len(train_loader)
    warmup_steps  = min(500, total_steps // 20)

    decay_params     = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{'params': decay_params, 'weight_decay': 0.1},
         {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=lr, betas=(0.9, 0.95)
    )
    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
    model.to(DEVICE)

    all_metrics = {}
    with open('metrics.json', 'w') as f:
        json.dump(all_metrics, f)

    global_step = 0
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        for x, y in pbar:
            global_step += 1
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val * x.numel()
            total_tokens += x.numel()

            pbar.set_postfix({'loss': f"{loss_val:.4f}"})

            if global_step % 100 == 0:
                train_ppl_step = math.exp(loss_val)
                if f"epoch_{epoch}" not in all_metrics:
                    all_metrics[f"epoch_{epoch}"] = {"steps": {}}
                
                all_metrics[f"epoch_{epoch}"]["steps"][str(global_step)] = {
                    'train_loss': round(loss_val, 4),
                    'train_perplexity': round(train_ppl_step, 4)
                }
                with open('metrics.json', 'w') as f:
                    json.dump(all_metrics, f, indent=4)

        train_loss_epoch = total_loss / total_tokens
        train_ppl = math.exp(train_loss_epoch)
        val_loss, val_ppl = evaluate(model, val_loader)
        print(f"[Epoch {epoch}] train_loss={train_loss_epoch:.4f} train_perplexity={train_ppl:.2f} val_loss={val_loss:.4f} val_perplexity={val_ppl:.2f}")
        
        if f"epoch_{epoch}" not in all_metrics:
            all_metrics[f"epoch_{epoch}"] = {"steps": {}}
            
        all_metrics[f"epoch_{epoch}"].update({
            'val_loss': round(val_loss, 4),
            'val_perplexity': round(val_ppl, 4)
        })
        with open('metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=4)

@torch.no_grad()
def evaluate(model: GPTLanguageModel, val_loader: DataLoader) -> tuple[float, float]:
    model.eval()
    model.to(DEVICE)
    total_loss   = 0.0
    total_tokens = 0

    for x, y in tqdm(val_loader, desc="[Validation]"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        _, loss = model(x, y)
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()

    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


# ==========================================
# 6. Kaggle Environment Execution Block
# ==========================================
if __name__ == '__main__':
    # Tokenizer & Data Overrides
    corpus_dir = CORPUS_DIR
    vocab_sz = VOCAB_SIZE
    model_prefix = MODEL_PREFIX
    force_retrain_tokenizer = False

    # Training Hyperparameters
    epochs = 15
    batch_size = 16
    lr = 3e-4
    context = 256
    stride = 128
    
    embed_dim = 512
    num_heads = 8
    num_layers = 8
    ffn_dim = 2048
    dropout = 0.1

    save_path = 'minigpt.pt'
    load_path = None
    gen_only = False
    prompt = 'यह एक'
    gen_tokens = 50
    temperature = 1.0
    top_k = 50

    print(f"Device: {DEVICE}")

    # 1. Tokenizer Setup
    if force_retrain_tokenizer or not os.path.exists(f"{model_prefix}.model"):
        print("Tokenizer model not found or forced. Training...")
        os.makedirs(corpus_dir, exist_ok=True)
        train_tokenizer(corpus_dir, vocab_sz, model_prefix)
    
    VSZ = vocab_size()
    print(f"Vocabulary size: {VSZ}")

    # 2. Model Setup
    model = GPTLanguageModel(
        vocab_sz=VSZ, context_length=context, embed_dim=embed_dim,
        num_heads=num_heads, num_layers=num_layers, ffn_dim=ffn_dim,
        dropout=dropout
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    if load_path and os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path, map_location=DEVICE))
        print(f"Loaded checkpoint from {load_path}")

    # 3. Train Setup
    if not gen_only:
        train_loader, val_loader = get_corpus_loaders(
            context, stride, batch_size, corpus_dir=corpus_dir
        )
        
        train(model, train_loader, val_loader, epochs, lr)
        
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        # Validation Evaluation
        _, val_loader2 = get_corpus_loaders(
            context, stride, batch_size, corpus_dir=corpus_dir
        )
        val_loss, val_ppl = evaluate(model, val_loader2)
        print(f"Final Validation Loss: {val_loss:.4f} | Perplexity: {val_ppl:.4f}")

    # 4. Generation
    print(f"\n--- Generating {gen_tokens} tokens (Prompt: '{prompt}') ---")
    prompt_ids = encode(prompt, add_bos=True)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    generated = model.generate(prompt_tensor, gen_tokens, temperature, top_k)
    generated_text = decode(generated[0].tolist())
    print(f"Generated text:\n{generated_text}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(generated_text)
    print(f"Generated text saved to {OUTPUT_FILE}.")
