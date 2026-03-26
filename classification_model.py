import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from language_model import MiniGPT, DEVICE, CONTEXT_LENGTH, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FFN_DIM
DROPOUT = 0.3
from classification_data import get_classification_loaders
from tokenizer import vocab_size as get_vocab_size, decode

# Hyperparameters
NUM_CLASSES = 3
LEARNING_RATE = 2e-5  # Lower LR for fine-tuning
BATCH_SIZE = 32
NUM_EPOCHS = 5
MAX_SEQ_LEN = 256

class GPTClassifier(nn.Module):
    """
    Classifier using the MiniGPT backbone.
    Extracts features of the last token in the sequence for classification.
    """
    def __init__(self, vocab_sz: int = None, context_length: int = CONTEXT_LENGTH,
                 embed_dim: int = EMBED_DIM, num_heads: int = NUM_HEADS,
                 num_layers: int = NUM_LAYERS, ffn_dim: int = FFN_DIM,
                 dropout: float = DROPOUT, num_classes: int = NUM_CLASSES):
        super().__init__()
        if vocab_sz is None:
            vocab_sz = get_vocab_size()
        
        self.backbone = MiniGPT(vocab_sz, context_length, embed_dim, num_heads, num_layers, ffn_dim, dropout)
        self.clf_head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        # idx shape: (B, T)
        hidden = self.backbone(idx)        # (B, T, C)
        
        # Sequence Alignment: Extract last REAL token's features (first non-pad from the right)
        # For Right Padding, find the index of the last non-PAD token
        non_pad_mask = (idx != 0) # Assuming 0 is PAD_ID (PAD_ID=0)
        lengths = non_pad_mask.sum(dim=1) - 1
        lengths = torch.clamp(lengths, min=0) # Handle empty sequences if any
        
        # Gather the features at the last non-pad positions
        # hidden has shape (B, T, C), we want (B, C)
        batch_indices = torch.arange(idx.size(0), device=idx.device)
        last_hidden = hidden[batch_indices, lengths] # (B, C)
        
        logits = self.clf_head(last_hidden) # (B, num_classes)
        
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss


def train_classifier(model, train_loader, val_loader, epochs, lr, save_path):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Add scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.to(DEVICE)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})
        
        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        val_loss, val_acc = evaluate_classifier(model, val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
    torch.save(model.state_dict(), save_path)
    print(f"Classifier saved to {save_path}")
    return history


@torch.no_grad()
def evaluate_classifier(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, loss = model(x, y)
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    return total_loss / len(loader), correct / total


def save_plots(history, output_dir='plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Loss Curve
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    
    # Accuracy Curve
    plt.figure()
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'accuracy_curve.png'))
    print(f"Plots saved to {output_dir}/")


@torch.no_grad()
def show_correct_samples(model, loader, num_samples=5):
    model.eval()
    correct_samples = []
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    
    for x, y in loader:
        x_dev, y_dev = x.to(DEVICE), y.to(DEVICE)
        logits, _ = model(x_dev, y_dev)
        preds = torch.argmax(logits, dim=1)
        
        for i in range(x.size(0)):
            if preds[i] == y_dev[i]:
                text = decode(x[i].tolist())
                correct_samples.append({
                    'text': text,
                    'prediction': label_map[int(preds[i])],
                    'label': label_map[int(y_dev[i])]
                })
            if len(correct_samples) >= num_samples:
                break
        if len(correct_samples) >= num_samples:
            break
            
    print("\n--- 5 Correct Predictions ---")
    with open('correct_samples.txt', 'w', encoding='utf-8') as f:
        for i, sample in enumerate(correct_samples, 1):
            line = f"Sample {i}:\nText: {sample['text'][:200]}...\nPred: {sample['prediction']}\nLabel: {sample['label']}\n"
            print(line)
            f.write(line + "\n")
    print("Correct samples saved to correct_samples.txt")


def main():
    parser = argparse.ArgumentParser(description="GPT Sentiment Classifier")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--context', type=int, default=CONTEXT_LENGTH)
    parser.add_argument('--embed_dim', type=int, default=EMBED_DIM)
    parser.add_argument('--num_heads', type=int, default=NUM_HEADS)
    parser.add_argument('--num_layers', type=int, default=NUM_LAYERS)
    parser.add_argument('--ffn_dim', type=int, default=FFN_DIM)
    parser.add_argument('--dropout', type=float, default=DROPOUT)
    parser.add_argument('--load_lm', type=str, default='minigpt.pt', help='Pre-trained LM weights')
    parser.add_argument('--save', type=str, default='gpt_classifier.pt')
    args = parser.parse_args()

    # 1. Load Data
    train_loader, val_loader = get_classification_loaders(max_length=args.context, batch_size=args.batch_size)

    # 2. Build Model
    model = GPTClassifier(
        context_length=args.context,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout
    )
    
    # 3. Load Backbone Weights
    if os.path.exists(args.load_lm):
        print(f"Loading pre-trained backbone from {args.load_lm}...")
        lm_state = torch.load(args.load_lm, map_location=DEVICE)
        # Extract only backbone weights
        backbone_state = {k.replace('backbone.', ''): v for k, v in lm_state.items() if k.startswith('backbone.')}
        model.backbone.load_state_dict(backbone_state)
    else:
        print("Warning: No pre-trained backbone found. Training from scratch.")

    # 4. Train
    print("Starting classification training...")
    history = train_classifier(model, train_loader, val_loader, args.epochs, args.lr, args.save)

    # 5. Save Plots & Sample Correct Predictions
    save_plots(history)
    show_correct_samples(model, val_loader)

if __name__ == '__main__':
    main()
