import torch
from torch.utils.data import DataLoader

from sequence_dataset import SequenceDataset

import time


def extract_embeddings(model, sequences, embedding_size, batch_size):
    batch_size = 100
    print(batch_size)
    start = time.time()
    dataloader = DataLoader(SequenceDataset(sequences), batch_size=batch_size)
    model.eval()
    extracted_embeddings = torch.empty((len(sequences), embedding_size))
    with torch.no_grad():
        for b, batch in enumerate(dataloader):
            embeddings = model(batch, repr_layers=[33])["representations"][33]
            sequence_embeddings = embeddings.narrow(
                1, 1, embeddings.size()[1] - 2
            ).mean(1)
            extracted_embeddings[b * batch_size : (b + 1) * batch_size, :] = (
                sequence_embeddings
            )
    extracted_embeddings = extracted_embeddings.cpu()
    print(time.time() - start)
    return extracted_embeddings
