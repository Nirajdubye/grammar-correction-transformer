import os
import math
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

from inference import SinusoidalPositionalEncoding, TransformerModel, MAX_LEN


# ---------------------------------------------------------------------------
# Dataset & Collate Function
# ---------------------------------------------------------------------------
class GECDataset(Dataset):
    def __init__(self, csv_path: str, input_col: str = "input", output_col: str = "output"):
        self.df = pd.read_csv(csv_path)
        self.input_col = input_col if input_col in self.df.columns else self.df.columns[0]
        self.output_col = output_col if output_col in self.df.columns else self.df.columns[1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        inp = str(self.df.iloc[idx][self.input_col]).strip()
        out = str(self.df.iloc[idx][self.output_col]).strip()
        return inp, out


def create_collate_fn(tokenizer, pad_idx, sos_idx, eos_idx, max_len=MAX_LEN):
    def encode(text):
        return [sos_idx] + tokenizer.encode(text).ids[:max_len - 2] + [eos_idx]

    def collate_fn(batch):
        src, tgt = zip(*batch)
        src = pad_sequence(
            [torch.tensor(encode(s)) for s in src],
            batch_first=True,
            padding_value=pad_idx
        )[:, :max_len]

        tgt = pad_sequence(
            [torch.tensor(encode(t)) for t in tgt],
            batch_first=True,
            padding_value=pad_idx
        )[:, :max_len]

        return src, tgt

    return collate_fn


# ---------------------------------------------------------------------------
# Validation Function
# ---------------------------------------------------------------------------
def evaluate_epoch(model, loader, criterion, vocab_size, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            with amp.autocast():
                output = model(src, tgt[:, :-1])
                loss = criterion(
                    output.reshape(-1, vocab_size),
                    tgt[:, 1:].reshape(-1)
                )
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------------------------
def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Training on device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 1. Load Tokenizer
    tokenizer = ByteLevelBPETokenizer(
        os.path.join(args.tokenizer_dir, "vocab.json"),
        os.path.join(args.tokenizer_dir, "merges.txt")
    )
    pad_idx = tokenizer.token_to_id("<pad>")
    sos_idx = tokenizer.token_to_id("<sos>")
    eos_idx = tokenizer.token_to_id("<eos>")
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary Size: {vocab_size}")

    collate_fn = create_collate_fn(tokenizer, pad_idx, sos_idx, eos_idx, max_len=args.max_len)

    # 2. Data Preparation
    print(f"Loading dataset from: {args.data_csv}")
    dataset = GECDataset(args.data_csv)
    train_size = int((1.0 - args.val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 3. Model & Loss setup
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers
    ).to(device)

    # Upweight EOS to discourage infinite generation
    weights = torch.ones(vocab_size).to(device)
    weights[eos_idx] = args.eos_weight
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, weight=weights)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scaler = amp.GradScaler()
    writer = SummaryWriter(log_dir=args.log_dir)

    # 4. Optional Resume
    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming training from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)

    # 5. Training Loop
    print("\nStarting Training...")
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{start_epoch + args.epochs}")

        for step, (src, tgt) in enumerate(pbar):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()

            with amp.autocast():
                output = model(src, tgt[:, :-1])
                loss = criterion(
                    output.reshape(-1, vocab_size),
                    tgt[:, 1:].reshape(-1)
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            global_step += 1

            if global_step % args.log_every == 0:
                writer.add_scalar("Loss/train_step", loss.item(), global_step)
                pbar.set_postfix({"step_loss": f"{loss.item():.4f}"})

            # Checkpoint per N steps
            if global_step % args.save_every == 0:
                step_ckpt = os.path.join(args.checkpoint_dir, f"step_{global_step}.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step
                }, step_ckpt)
                print(f"\nSaved step checkpoint: {step_ckpt}")

        avg_train_loss = total_train_loss / len(train_loader)
        val_loss = evaluate_epoch(model, val_loader, criterion, vocab_size, device)

        print(f"\n--- Epoch {epoch + 1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} ---")
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch + 1)
        writer.add_scalar("Loss/val_epoch", val_loss, epoch + 1)

        # Save Epoch Checkpoint
        epoch_ckpt = os.path.join(args.checkpoint_dir, f"epoch_{epoch + 1}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss
        }, epoch_ckpt)
        print(f"Saved epoch checkpoint: {epoch_ckpt}")

    writer.close()
    print("Training finished successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GEC Transformer Model")
    parser.add_argument("--data_csv", type=str, default="data/train.csv", help="Path to input dataset CSV")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer", help="Directory with vocab.json & merges.txt")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--log_dir", type=str, default="runs", help="TensorBoard log directory")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing checkpoint to resume")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--max_len", type=int, default=MAX_LEN)
    parser.add_argument("--eos_weight", type=float, default=5.0)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--save_every", type=int, default=25000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(args)
