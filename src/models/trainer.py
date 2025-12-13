import torch
from .classifier import FoodImageDataset
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup
import random

def fine_tune(
    model_wrapper,
    data_dir,
    val_ratio=0.1,
    epochs=3,
    batch_size=16,
    lr=5e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    save_model = False
):

    # Load dataset
    full_ds = FoodImageDataset(data_dir, model_wrapper.processor, model_wrapper.labels)
    n = len(full_ds)
    indices = list(range(n))
    random.shuffle(indices)

    # train/val plit
    split = int(n * (1 - val_ratio))
    train_idx, val_idx = indices[:split], indices[split:]

    train_ds = Subset(full_ds, train_idx)
    val_ds = Subset(full_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # setup 
    model = model_wrapper.model
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    # Main training loop
    correct, total = 0, 0
    training_history = {"loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(epochs):
        running_loss = 0.0

        for pixel_values, labels, _ in train_loader:
            pixel_values = pixel_values.to(model_wrapper.device)
            labels = labels.to(model_wrapper.device)

            logits = model(pixel_values=pixel_values).logits
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            accuracy = correct / total

        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss / len(train_loader):.4f} - Accuracy: {accuracy:.4f}")
        training_history["loss"].append(running_loss / len(train_loader))
        training_history["train_acc"].append(accuracy)

        # validation after each epoch
        val_acc = validate(model, val_loader, model_wrapper.device)
        training_history["val_acc"].append(val_acc)

    # save model
    if save_model:
        model.save_pretrained("finetuned_food101_siglip")
        model_wrapper.processor.save_pretrained("finetuned_food101_siglip")
        print("Model saved to: finetuned_food101_siglip")

    return training_history

def validate(model, loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for pixel_values, labels, _ in loader:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            logits = model(pixel_values=pixel_values).logits
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Validation accuracy: {acc:.4f}")
    model.train()
    return acc