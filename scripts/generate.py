
import torch
import tiktoken
import time

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transformer_models.gpt2 import GPT2, GPT2Config
from scripts.train import load_data
from tokenizer.character_tokenizer import CharacterTokenizer

def main():
    tokenizer = tiktoken.get_encoding("gpt2")

    config = GPT2Config()
    config.vocab_size = tokenizer.n_vocab

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    model = GPT2(config).to(device)
    model.load_state_dict(torch.load("checkpoint/gpt2_step2000.pt", map_location=device))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")

    model.eval()
    with torch.no_grad():
        seed = "The"
        query = torch.tensor(tokenizer.encode(seed), dtype=torch.long, device=device)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        torch.manual_seed(0)
        t0 = time.perf_counter()
        out1 = model.generate(query, max_new_tokens=200, temperature=0.8, top_k=10, use_cache=False)     #without kv cache
        t1 = time.perf_counter()

        torch.manual_seed(0)
        t2 = time.perf_counter()
        out2 = model.generate(query, max_new_tokens=200, temperature=0.8, top_k=10, use_cache=True)  #with kv cache
        t3 = time.perf_counter()

        result1 = tokenizer.decode(out1[0].tolist())
        result2 = tokenizer.decode(out2[0].tolist())
        print(f"Without cache : {t1-t0:.3f}s")
        print(f"With cache: {t3-t2:.3f}s")
        print(f"Speedup: {(t1-t0)/(t3-t2):.2f}")

        print(f"\n\n---Generated Text 1---\n{result1}")
        print(f"\n\n---Generated Text 2---\n{result2}")


if __name__=="__main__":
    main()

