import torch.nn as nn

config = {
    "n_layers":6,
    "n_heads":6,
    "d_model":384,
    "vocab_size":50257,
    "block_size":256,
    "expansion_factor":4,
    "dropout":0.1,
    "padding_token":0
}

class GPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["d_model"])

    
