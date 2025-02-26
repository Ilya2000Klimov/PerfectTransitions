# train_lstm.py
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
    parser.add_argument("--resume_if_checkpoint_exists", type=bool, default=False)
    # We won't specify fixed hyperparams here since W&B sweeps will override them
    return parser.parse_args()

def train(config=None):
    """
    The main training function called by W&B sweeps.
    config: dict of hyperparameters from wandb.config
    """
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Initialize W&B
    wandb.init(project="BEATs-LSTM-Transitions", config=config)
    config = wandb.config  # Pull in hyperparams from wandb

    # 2) Create dataset & DataLoader
    train_dataset = TransitionsDataset(config.embeddings_dir, split="train", overlap_frames=config.num_frames)
    val_dataset   = TransitionsDataset(config.embeddings_dir, split="val", overlap_frames=config.num_frames)
    test_dataset  = TransitionsDataset(config.embeddings_dir, split="test", overlap_frames=config.num_frames)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 3) Define Model, Loss, Optimizer
    model = BiLSTMTransitionModel(
        input_dim=train_dataset.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.lstm_layers,
        dropout=0.2
    ).to(device)

    triplet_loss_fn = nn.TripletMarginLoss(margin=config.margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    scaler = torch.amp.GradScaler(device="cuda")  # Mixed precision

    # 4) Check for checkpoint
    start_epoch = 0
    checkpoint_file = os.path.join(config.checkpoint_dir, "lstm_checkpoint.pt")
    if config.resume_if_checkpoint_exists and os.path.exists(checkpoint_file):
        ckpt = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] Resumed training from epoch {start_epoch}")

    best_val_loss = float("inf")
    epochs_no_improve = 0

    # 5) Training loop
    for epoch in range(start_epoch, config.max_epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            anchor, positive, negative = batch
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                anchor_out = model(anchor)
                positive_out = model(positive)
                negative_out = model(negative)
                loss = triplet_loss_fn(anchor_out, positive_out, negative_out)

            scaler.scale(loss).backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                anchor, positive, negative = batch
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                anchor_out = model(anchor)
                positive_out = model(positive)
                negative_out = model(negative)
                val_loss = triplet_loss_fn(anchor_out, positive_out, negative_out)
                total_val_loss += val_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch [{epoch}/{config.max_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save checkpoint every 5 epochs or if improvement
        if ((epoch + 1) % 5 == 0) or (avg_val_loss == best_val_loss):
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
            }, checkpoint_file)
            wandb.save(checkpoint_file, base_path=config.checkpoint_dir)
            print(f"Checkpoint saved at epoch {epoch}")

        if epochs_no_improve >= config.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Final test evaluation
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            anchor, positive, negative = batch
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            anchor_out = model(anchor)
            positive_out = model(positive)
            negative_out = model(negative)
            test_loss = triplet_loss_fn(anchor_out, positive_out, negative_out)
            total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.4f}")
    wandb.log({"test_loss": avg_test_loss})
    wandb.finish()

def main():
    args = parse_args()
    # We'll pass a dictionary of hyperparams to train()
    base_config = {
        "embeddings_dir": args.embeddings_dir,
        "checkpoint_dir": args.checkpoint_dir,
        "batch_size": 32,
        "margin": 0.3,
        "lstm_layers": 2,
        "hidden_dim": 128,
        "lr": 1e-3,
        "max_epochs": 50,
        "patience": 5,
        "resume_if_checkpoint_exists": args.resume_if_checkpoint_exists
    }

    train(base_config)

if __name__ == "__main__":
    main()
