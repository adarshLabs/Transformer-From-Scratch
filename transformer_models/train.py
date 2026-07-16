import math

import matplotlib.pyplot as plt
import torch

from transformer_models.gpt2 import GPT2, GPT2Config

MAX_STEPS = 501
WARMUP_STEPS = 50
MAX_LR = 3e-4
MIN_LR = 3e-5
MAX_NORM = 1.0
BATCH_SIZE = 8


def get_batch(data, block_size, batch_size, device):
    indices = torch.randint(len(data) - 1 - block_size, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in indices])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in indices])

    return x.to(device), y.to(device)


def get_lr(step, warmup_steps, max_lr, min_lr, max_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    if step > max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(progress * math.pi))


def train(model, data, block_size, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1)
    lrs = []
    losses = []

    for step in range(MAX_STEPS):
        lr = get_lr(step, WARMUP_STEPS, MAX_LR, MIN_LR, MAX_STEPS)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(data, block_size, BATCH_SIZE, device)
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
        optimizer.step()

        lrs.append(float(lr))
        losses.append(float(loss.item()))

        if step % 50 == 0:
            print(f"Loss at step {step} with lr {lr}: {loss.item():.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(lrs)), lrs, label="learning rate")
    plt.xlabel("step")
    plt.ylabel("learning rate")
    plt.title("Learning rate schedule")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(range(len(losses)), losses, label="loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.legend()
    plt.show()


def main():
    config = GPT2Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GPT2(config).to(device)
    data = torch.randint(1, config.vocab_size, (10 * config.block_size,)).to(device)
    train(model, data, config.block_size, device)


if __name__ == "__main__":
    main()
