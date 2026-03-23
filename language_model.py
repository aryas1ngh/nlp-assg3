import math
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from tokenizer import vocab_size as get_vocab_size, decode, encode, BOS_ID
from corpus_data import get_corpus_loaders

# Hyperparameters
CONTEXT_LENGTH = 256
EMBED_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 8
FFN_DIM = 2048
DROPOUT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 6e-4
NUM_EPOCHS = 5
STRIDE = 64
OUTPUT_FILE = 'generated_text.txt'

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ── 1. Multi-Head Attention (with causal mask) ────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Key, Query, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape          # batch, seq-len, embed_dim

        # Project and split into heads → (B, num_heads, T, head_dim)
        def split_heads(t):
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(self.q_proj(x))   # (B, H, T, head_dim)
        K = split_heads(self.k_proj(x))
        V = split_heads(self.v_proj(x))

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale   # (B, H, T, T)

        # Causal mask: positions can only attend to previous (and current) positions
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn_scores = attn_scores.masked_fill(~causal_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum over values, then merge heads
        out = torch.matmul(attn_weights, V)         # (B, H, T, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, embed_dim)
        return self.out_proj(out)


# --- Feed-Forward Network ---
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


# --- Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual connections (GPT-2 style)
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# --- MiniGPT Backbone ---
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

        # Weight tying: token embeddings ↔ LM head is done in GPTLanguageModel
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


# --- Language Model Head ---
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_sz: int = None, context_length: int = CONTEXT_LENGTH,
                 embed_dim: int = EMBED_DIM, num_heads: int = NUM_HEADS,
                 num_layers: int = NUM_LAYERS, ffn_dim: int = FFN_DIM,
                 dropout: float = DROPOUT):
        super().__init__()
        if vocab_sz is None:
            vocab_sz = get_vocab_size()

        self.backbone = MiniGPT(vocab_sz, context_length, embed_dim, num_heads, num_layers, ffn_dim, dropout)
        self.lm_head = nn.Linear(embed_dim, vocab_sz, bias=False)
        # Weight tying: LM head shares the token-embedding matrix
        self.lm_head.weight = self.backbone.token_emb.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """Pass input through the backbone and LM head; return (logits, loss)."""
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
        """Autoregressively sample tokens from the model."""
        self.eval()
        ctx_len = self.backbone.context_length
        idx = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # Crop context to fit the model's window
            idx_cond = idx[:, -ctx_len:]
            logits, _ = self(idx_cond)        # (1, T, V)
            logits = logits[:, -1, :]         # last-token logits (1, V)

            # Temperature scaling
            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx

# LR Scheduler
def get_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))  # floor at 5% of peak LR
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# --- Training Loop ---
def train(model: GPTLanguageModel, train_loader: DataLoader, val_loader: DataLoader,
          num_epochs: int = NUM_EPOCHS, lr: float = LEARNING_RATE):
    total_steps   = num_epochs * len(train_loader)
    warmup_steps  = min(500, total_steps // 20)   # ~5% warmup

    # Separate weight decay: don't decay biases, LayerNorm, embeddings
    decay_params     = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params  = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{'params': decay_params, 'weight_decay': 0.1},
         {'params': no_decay_params, 'weight_decay': 0.0}],
        lr=lr, betas=(0.9, 0.95)   # ← beta2=0.95 is standard for LMs
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

        # Epoch summary
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
    """Return validation loss and perplexity."""
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


# --- CLI Entry Point ---
def main():
    parser = argparse.ArgumentParser(description="MiniGPT Hindi Language Model")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--context', type=int, default=CONTEXT_LENGTH)
    parser.add_argument('--embed_dim', type=int, default=EMBED_DIM)
    parser.add_argument('--num_heads', type=int, default=NUM_HEADS)
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS)
    parser.add_argument('--ffn_dim', type=int, default=FFN_DIM)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--stride', type=int, default=STRIDE)
    parser.add_argument('--save', type=str, default='minigpt.pt', help='Save model path')
    parser.add_argument('--load', type=str, default=None, help='Load model path')
    parser.add_argument('--gen_only', action='store_true', help='Generate text only')
    parser.add_argument('--prompt', type=str, default='यह एक', help='Hindi prompt')
    parser.add_argument('--gen_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=50)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    vocab_sz = get_vocab_size()
    print(f"Vocabulary size: {vocab_sz}")

    model = GPTLanguageModel(
        vocab_sz=vocab_sz, context_length=args.context, embed_dim=args.embed_dim,
        num_heads=args.num_heads, num_layers=args.num_layers, ffn_dim=args.ffn_dim,
        dropout=args.dropout
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=DEVICE))
        print(f"Loaded checkpoint from {args.load}")

    if not args.gen_only:
        train_loader, val_loader = get_corpus_loaders(args.context, args.stride, args.batch_size)
        train(model, train_loader, val_loader, args.epochs, args.lr)
        torch.save(model.state_dict(), args.save)
        print(f"Model saved to {args.save}")

        _, val_loader = get_corpus_loaders(args.context, args.stride, args.batch_size)
        val_loss, val_ppl = evaluate(model, val_loader)
        print(f"Final Validation Loss: {val_loss:.4f} | Final Validation Perplexity: {val_ppl:.4f}")

    print(f"\n--- Generating {args.gen_tokens} tokens (prompt: '{args.prompt}') ---")
    prompt_ids = encode(args.prompt, add_bos=True)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    generated = model.generate(prompt_tensor, args.gen_tokens, args.temperature, args.top_k)
    generated_text = decode(generated[0].tolist())
    print(f"Generated text:\n{generated_text}")
    with open(OUTPUT_FILE, 'w') as f:
        f.write(generated_text)
    print(f"Generated text saved to {OUTPUT_FILE}.")


if __name__ == '__main__':
    main()
