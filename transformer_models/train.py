from gpt2 import GPT2, GPT2Config
import torch

# Training hyperparameters
MAX_STEPS = 250
BATCH_SIZE = 8
LEARNING_RATE = 1e-4


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data)-1- block_size, (batch_size,))

    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)

def train(model, data, block_size, device):
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for step in range(MAX_STEPS):
        x, y = get_batch(data, block_size, BATCH_SIZE, device)
        logits, loss = model(x, y)

        optimiser.zero_grad()
        loss.backward()

        optimiser.step()

        if step % 10 == 0:
            print(f"Loss at step {step}: {loss.item():.4f}")


def main():
    config = GPT2Config()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    model = GPT2(config).to(device)
    data = torch.randint(1, config.vocab_size, (2*config.block_size,)).to(device)
    train(model, data, config.block_size, device)


if __name__=="__main__":
    main()