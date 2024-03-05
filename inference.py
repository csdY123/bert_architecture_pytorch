import torch
from model import BERT
def main():
    # Example of using BERT model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    num_layers = 2
    vocab_size = 10000  # This should be adjusted to your dataset.
    embed_size = 256
    heads = 8
    forward_expansion = 4
    dropout = 0.1
    max_length = 100  # Max length of a sentence

    model = BERT(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_layers=num_layers,
        heads=heads,
        device=device,
        forward_expansion=forward_expansion,
        dropout=dropout,
        max_length=max_length,
    ).to(device)

    # Dummy input for testing
    x = torch.tensor([[1, 2, 3, 4, 5, 0, 0],[1, 2, 3, 4, 5, 0, 0]]).to(device)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0],[1, 1, 1, 1, 1, 0, 0]]).to(device)

    output = model(x, mask=mask)
    print(output.shape)  # (batch_size, seq_length, embed_size)

main()