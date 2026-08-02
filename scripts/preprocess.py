import argparse
from pathlib import Path
from datasets import load_dataset
import numpy as np
import tiktoken
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="tiny_shakespeare",
        choices=["tiny_shakespeare", "tiny_stories"],
    )
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--tokenizer", default="gpt2")

    return parser.parse_args()


def load_text_data(dataset_name):
    if dataset_name == "tiny_shakespeare":
        path = Path("data/raw") / "tiny_shakespeare.txt"

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        split = int(0.9 * len(text))
        return text[:split], text[split:]

    if dataset_name == "tiny_stories":
        dataset = load_dataset("roneneldan/TinyStories")
        train_text = "\n".join(sample["text"] for sample in dataset["train"])
        val_text = "\n".join(sample["text"] for sample in dataset["validation"])
        return train_text, val_text

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def encode_dataset(text, tokenizer):
    ids = tokenizer.encode_ordinary(text)
    ids.append(tokenizer.eot_token)
    return ids


def save_bin(ids, path):
    memmap = np.memmap(path, dtype=np.uint16, mode="w+", shape=(len(ids),))
    memmap[:] = np.array(ids, dtype=np.uint16)
    memmap.flush()


def main():
    args = parse_args()
    tokenizer = tiktoken.get_encoding(args.tokenizer)
    train_text, val_text = load_text_data(args.dataset)

    train_ids = encode_dataset(train_text, tokenizer)
    val_ids = encode_dataset(val_text, tokenizer)

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    save_bin(train_ids, output_dir / "train.bin")
    save_bin(val_ids, output_dir / "val.bin")

    meta = {
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.n_vocab,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
    }

    with open(output_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("Done.")
    print(f"Train tokens : {len(train_ids):,}")
    print(f"Val tokens   : {len(val_ids):,}")
    print(f"Saved to     : {output_dir}")


if __name__ == "__main__":
    main()
