import torch
from torch.utils.data import DataLoader

from sequence_dataset import SequenceDataset

def train(model, sequences, mask_positions, mask_token, n_tokens, device):
    # TODO ? different optimizer
    # TODO optimize learning rate

    # TODO optimize batch_size
    # TODO optimize copying to cuda device (all data beforehand or each batch individually)

    # TODO optimize number of epochs

    if type(mask_positions) != torch.Tensor:
        mask_positions = torch.tensor(mask_positions)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001)
    loss_function = torch.nn.CrossEntropyLoss()
    dataloader = DataLoader(SequenceDataset(sequences.to(device)), batch_size=10)
    epochs = 1

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            input = batch.clone()
            input[:, mask_positions] = mask_token
            logits = model(input)['logits']

            logits = logits[:, mask_positions, :]
            batch = batch[:, mask_positions]
            loss = loss_function(logits.view(-1, n_tokens), batch.view(-1))

            epoch_loss += loss
            loss.backward()
            optimizer.step()

        print("Epoch: {} -> mean loss during epoch: {}".format(epoch+1, epoch_loss/(len(dataloader))))
