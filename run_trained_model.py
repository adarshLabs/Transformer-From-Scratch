
from transformer_models.gpt2 import GPT2, GPT2Config
from train import load_data
from tokenizer.charactor_tokenizer import CharacterTokenizer
import torch
import tiktoken
import time


def main():
    tokenizer = tiktoken.get_encoding("gpt2")

    config = GPT2Config()
    config.vocab_size = tokenizer.n_vocab
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = GPT2(config).to(device)
    model.load_state_dict(torch.load("checkpoint/gpt2_step500.pt", map_location=device))

    model.eval()
    with torch.no_grad():
        seed = "The"
        query = torch.tensor(tokenizer.encode(seed), dtype=torch.long, device=device)
        if query.dim() == 1:
            query = query.unsqueeze(0)

        t0 = time.perf_counter()
        out1 = model.generate(query, max_new_tokens=200, temperature=0.8, top_k=10, use_cache=False)     #without kv cache
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        out2 = model.generate(query, max_new_tokens=200, temperature=0.8, top_k=10, use_cache=True)  #with kv cache
        t3 = time.perf_counter()

        result1 = tokenizer.decode(out2[0].tolist())
        result2 = tokenizer.decode(out2[0].tolist())
        print(f"Without cache : {t1-t0:.3f}s")
        print(f"With cache: {t3-t2:.3f}s")
        print(f"Speedup: {(t1-t0)/(t3-t2):.2f}")
        print(f"Answer 1: {result1}")
        print(f"Answer 2: {result2}")


if __name__=="__main__":
    main()

