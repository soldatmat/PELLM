import torch

EMBEDDING_LENGTH = 1024


def extract_embedding(sequence, model):
    """
    [sequence] needs to be tokenized
    BOS and EOS tokens need to be removed form [sequence]
    Consider removing C- and N- terminus tokens from [sequence]
    [sequence] needs to be on the same device as [model]
    """
    with torch.no_grad():
        out = model(
            sequence,
            output_hidden_states=True,
        )

    # mean over residues to get per-sequence embedding
    embedding = out.hidden_states[-1].mean(0)
    return embedding


def extract_embeddings(sequences, model):
    embeddings = torch.empty((len(sequences), EMBEDDING_LENGTH), dtype=model.dtype)
    for s in range(len(sequences)):
        embeddings[s, :] = extract_embedding(sequences[s], model)
    return embeddings
