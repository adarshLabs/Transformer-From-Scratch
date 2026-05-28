
# pyrefly: ignore [missing-import]
import torch
from transformer_models.transformer_encoder import TransformerEncoder


def main():
    vocab_size = 1000
    embed_dim = 64
    num_heads = 8
    num_layers = 4
    positional_encoding_type="rope"
    batch_size = 2
    seq_len = 10

    input_ids = torch.randint(
        low=0,
        high = vocab_size,
        size=(batch_size, seq_len)
    )

    model = TransformerEncoder(
        embed_dim=embed_dim,num_layers=num_layers,num_heads=num_heads,
        vocab_size=vocab_size,positional_encoding_type=positional_encoding_type
        )
    
    output = model(input_ids)

    expected_shape = (batch_size, seq_len, embed_dim)

    assert output.shape==expected_shape



    assert not torch.isnan(output).any()

    print("\nTransformer Encoder Test")

    print(f"Input Shape  : {input_ids.shape}")

    print(f"Output Shape : {output.shape}")

    print("\nEnd-to-end transformer encoder test passed.")


if __name__=="__main__":
    main()

