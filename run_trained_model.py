
from transformer_models.gpt2 import GPT2, GPT2Config
from train import load_data
from tokenizer.charactor_tokenizer import CharacterTokenizer
import torch
import tiktoken


def main():
    tokenizer = tiktoken.get_encoding("gpt2")

    config = GPT2Config()
    config.vocab_size = tokenizer.n_vocab
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = GPT2(config).to(device)
    model.load_state_dict(torch.load("checkpoint/gpt2_step500.pt", map_location=device))

    model.eval()
    with torch.no_grad():
        text = input("Write you Query: ")
        while text!="exit":
            query = torch.tensor(tokenizer.encode(text), dtype=torch.long, device=device)
            if query.dim() == 1:
                query = query.unsqueeze(0)
            out = model.generate(query, max_new_tokens=200, temperature=0.8, top_k=10)
            result = tokenizer.decode(out[0].tolist())
            print(result)

if __name__=="__main__":
    main()

