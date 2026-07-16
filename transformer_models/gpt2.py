import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from dataclasses import dataclass
from transformer_models.gpt_decoder_block import GPTDecoderBlock
from attention.masking import causal_mask, padding_mask, combined_mask
import torch.nn.functional as F
import math

@dataclass
class GPT2Config:
    n_layers: int = 6
    n_heads: int = 6
    d_model: int = 384
    vocab_size: int = 50257
    block_size: int = 256
    expansion_factor: int = 4
    dropout: float = 0.1
    padding_token: int = -11

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            GPTDecoderBlock(
                embed_dim=config.d_model,
                num_heads=config.n_heads, 
                expansion_factor=config.expansion_factor, 
                dropout=config.dropout
            ) 
            for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def _build_mask(self, input_ids):
        B, S = input_ids.shape
        device = input_ids.device

        causal = causal_mask(S, device=device)
        padding = padding_mask(input_ids, padding_token=self.config.padding_token)
        return combined_mask(causal, padding)
    
    def forward(self, input_ids, target=None):

        embedding = self.token_embedding(input_ids)
        x = self.drop(embedding)
        mask = self._build_mask(input_ids)

        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        #print(input_ids)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=0.8, top_k=50):

        self.eval()

        for _ in range(max_new_tokens):

            context = input_ids[:, -self.config.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :]/ temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, self.config.vocab_size))
                logits[logits < v[:,[-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

        return input_ids
    
def main():
    config = GPT2Config()
    model = GPT2(config)

    total = sum(p.numel() for p in model.parameters())

    print(f"Parameters: {total}:,")

    B, S = 2, 32

    input_ids = torch.randint(1, 50257, (B, S))
    input_ids[0, -3:] = 0                              #add padding to test
    targets = torch.randint(0, 50257, (B, S))

    logits, loss = model(input_ids, targets)

    print(f"Logits : {logits.shape}    expected ({B}, {S}, {config.vocab_size})")
    print(f"Loss   : {loss.item():.4f}  expected ~{math.log(config.vocab_size):.2f}")

    seed = torch.randint(1, 50257, (1, 5))
    out = model.generate(seed, max_new_tokens=20, temperature=0.8, top_k =50)

    print(f"Seed: {seed.shape}, Out: {out.shape}")
    print("All Checks Passed")


if __name__=="__main__":
    main()


