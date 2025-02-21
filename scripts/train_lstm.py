import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from torch.utils.data import DataLoader
from my_lstm_dataset import TransitionsDataset  # your custom dataset
from my_lstm_model import BiLSTMTransitionModel # your custom model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resume_if_checkpoint_exists", type=bool, default=False)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Initialize W&B (optional, if you want to track everything)
    wandb.init(project="BEATs-LSTM-Transitions", config=vars(args))

    # 2. Create dataset & dataloader
    train_dataset = TransitionsDataset(args.embeddings_dir, split="train")
    val_dataset   = TransitionsDataset(args.embeddings_dir, split="val")
    test_dataset  = TransitionsDataset(args.embeddings_dir, split="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Define Model, Loss, Optimizer
    model = BiLSTMTransitionModel(
        input_dim=train_dataset.input_dim, 
        hidden_dim=args.hidden_dim, 
        num_layers=args.lstm_layers, 
        dropout=0.2
    ).to(device)

    # Example: Triplet Loss + Cosine
    triplet_loss_fn = nn.TripletMarginLoss(margin=args.margin, p=2)  # or p=2, etc.
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Optional: Cosine annealing LR schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision

    # 4. Resume from checkpoint if available
    start_epoch = 0
    checkpoint_file = os.path.join(args.checkpoint_dir, "lstm_checkpoint.pt")
    if args.resume_if_checkpoint_exists and os.path.exists(checkpoint_file):
        ckpt = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] Resumed training from epoch {start_epoch}")

    best_val_loss = float("inf")
    epochs_no_improve = 0
    max_epochs = args.max_epochs

    # 5. Training loop
    for epoch in range(start_epoch, max_epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            anchor, positive, negative = batch  # shape [B, T, D] each
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(): 
                anchor_out = model(anchor)      # shape [B, embedding_dim]
                positive_out = model(positive)
                negative_out = model(negative)

                loss = triplet_loss_fn(anchor_out, positive_out, negative_out)

            scaler.scale(loss).backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        # LR scheduler step
        scheduler.step()

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                anchor, positive, negative = [b.to(device) for b in batch]
                anchor_out = model(anchor)
                positive_out = model(positive)
                negative_out = model(negative)
                val_loss = triplet_loss_fn(anchor_out, positive_out, negative_out)
                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch}/{max_epochs}] Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # Early Stopping Check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save checkpoint every 5 epochs or if improvement
        if (epoch + 1) % 5 == 0 or avg_val_loss == best_val_loss:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }, checkpoint_file)
            wandb.save(checkpoint_file)
            print(f"Checkpoint saved at epoch {epoch}")

        if epochs_no_improve >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # --- Final Test Evaluation (optional) ---
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            anchor, positive, negative = [b.to(device) for b in batch]
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            test_loss = triplet_loss_fn(anchor_out, positive_out, negative_out)
            total_test_loss += test_loss.item()
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.4f}")

    wandb.log({"test_loss": avg_test_loss})

if __name__ == "__main__":
    main()