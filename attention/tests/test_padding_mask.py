import torch

from attention.masking import padding_mask


def test_padding_mask_defaults_to_minus_one_padding_token():
    input_ids = torch.tensor([
        [1, 2, -1, -1],
        [3, 4, 5, -1],
    ], dtype=torch.long)

    mask = padding_mask(input_ids)

    expected = torch.tensor([
        [[[True, True, False, False]]],
        [[[True, True, True, False]]],
    ], dtype=torch.bool)

    assert mask.shape == expected.shape
    assert torch.equal(mask, expected)
