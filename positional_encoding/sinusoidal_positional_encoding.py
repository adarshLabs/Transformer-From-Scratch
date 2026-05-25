import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=5000):
        super().__init__()

        assert embed_dim % 2 == 0,  (
            "Embedding dimension must be even"
        )

        pe = torch.zeros(max_seq_len, embed_dim)

        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)     
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * - torch.log(torch.tensor(10000.0))/embed_dim) 

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) 

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)


    def forward(self, x):
        seq_len = x.size(1)
        assert seq_len <= self.pe.size(1), (
            "Sequence length exceeds "
            "maximum positional encoding length"
        )
        return x + self.pe[:, :seq_len]




def main():

    pe = SinusoidalPositionalEncoding(

        embed_dim=128,

        max_seq_len=100

    )

    encoding = pe.pe.squeeze(0)

    plt.imshow(
        encoding.numpy(),
        aspect="auto"
    )

    plt.xlabel("Embedding Dimension")
    plt.ylabel("Position")
    plt.title(
        "Sinusoidal Positional Encoding"
    )

    plt.colorbar()

    plt.show()


if __name__=="__main__":
    main()
