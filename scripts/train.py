import math
import os
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import tiktoken
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer_models.gpt2 import GPT2, GPT2Config
from tokenizer.character_tokenizer import CharacterTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", "--steps", dest="max_steps", type=int, default=2000)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_val_batches", type=int, default=50)
    parser.add_argument("--block_size", type=int, default=256)
    return parser.parse_args()


def load_data(path):
    
    with open(path, "r") as f:
        text = f.read()

    return text


def get_batch(data, block_size, batch_size, device):
    indices = torch.randint(len(data) - 1 - block_size, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in indices])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in indices])

    return x.to(device), y.to(device)


### if model can't overfit a single batch, then there should be .
def sanity_check(model, block_size, device):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
 
    # Fix ONE batch — sample once, reuse every step
    vocab_size = model.config.vocab_size
    x = torch.randint(0, vocab_size, (4, block_size), device=device)
    y = torch.randint(0, vocab_size, (4, block_size), device=device)
 
    loss_history = []
    for step in range(200):
        logits, loss, _ = model(x, y)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        loss_history.append(loss.item())
        if step % 20 == 0:
            print(f"  step {step:3d}:  loss {loss.item():.4f}")
 
    final_loss = loss_history[-1]
    status = "PASSED" if final_loss < 0.5 else "FAILED — check your implementation"
    print(f"\nFinal loss: {final_loss:.4f}  {status}\n")
    return loss_history


def get_lr(step, warmup_steps, max_lr, min_lr, max_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    if step > max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(progress * math.pi))


def train(
    model,
    data,
    block_size,
    decode,
    device,
    args,
):
    model.train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.max_lr, weight_decay=0.1)
    lr_history = []
    loss_history = []
    for step in range(args.max_steps + 1):
        lr = get_lr(step, args.warmup_steps, args.max_lr, args.min_lr, args.max_steps)
        for param_group in optimiser.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(data, block_size, args.batch_size, device)
        _, loss, _ = model(x, y)

        optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimiser.step()
        lr_history.append(float(lr))
        loss_history.append(float(loss.item()))
        if step % args.log_every == 0:
            print(f"step {step:4d} | loss {loss.item():.4f} | lr {lr:.6f} | grad_norm {grad_norm:.3f}")


        if step % 500==0 and step>0:
            os.makedirs("checkpoint", exist_ok=True)
            torch.save(model.state_dict(), f"checkpoint/gpt2_step{step}.pt")

            model.eval()
            seed = torch.zeros((1,1), dtype=torch.long, device=device)
            out = model.generate(seed, max_new_tokens=200, temperature=0.8, top_k=40)
            print(f"--- Text Generated at {step} ---")
            print(decode(out[0].tolist()))
            model.train()

    # plt.figure(figsize=(8, 4))
    # plt.plot(range(len(lr_history)), lr_history, label="learning rate")
    # plt.xlabel("step")
    # plt.ylabel("learning rate")
    # plt.title("Learning rate schedule")
    # plt.legend()
    # plt.show()

    # plt.figure(figsize=(8, 4))
    # plt.plot(range(len(loss_history)), loss_history, label="loss")
    # plt.xlabel("step")
    # plt.ylabel("loss")
    # plt.title("Training loss")
    # plt.legend()
    # plt.show()

def validation(model, val_data, block_size, device, args):
    model.eval()
    val_loss_history = []
    with torch.no_grad():
        for _ in range(args.num_val_batches):
            x, y = get_batch(val_data, block_size, args.batch_size, device=device)
            logits, loss, _ = model(x, y)

            val_loss_history.append(loss.item())


    val_loss = sum(val_loss_history)/len(val_loss_history)
    print("Average Validation Loss:", val_loss)

def main():

    config = GPT2Config()
    args = parse_args()
    config.block_size = args.block_size

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(device)
    print(f"Using device: {device}")
          
    # for local files
    path = "data/tiny_shakespeare.txt"
    text = load_data(path)

    # Character level tokenizer
    # tokenizer = CharacterTokenizer(text)   


    # Subword level BPE tokenizer   
    tokenizer = tiktoken.get_encoding('gpt2')               
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)



    config.vocab_size = tokenizer.n_vocab
    model = GPT2(config).to(device)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"Vocab Size : {config.vocab_size}")
    print(f"Total tokens: {len(data)}, Training tokens: {len(train_data)}, Validation tokens: {len(val_data)}")

    train(
        model,
        train_data,
        config.block_size,
        tokenizer.decode,
        device,
        args,
    )

    validation(
        model=model,
        val_data=val_data,
        block_size=config.block_size,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    main()
