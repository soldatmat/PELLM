import torch
from torch.utils.data import DataLoader

from variant_dataset import VariantDataset

def train(model, embeddings, fitness, device):
    # TODO batch_size: "a multiple of 1 to allow the model to choose the size of the batch size independently"
    # TODO optimize copying to cuda device (all data beforehand or each batch individually)

    if len(embeddings) == 0:
        return

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001) # AFP-DE (ADAM, 1e-5)
    loss_function = torch.nn.functional.l1_loss
    #loss_function = torch.nn.functional.mse_loss
    dataloader = DataLoader(VariantDataset(embeddings.to(device), fitness.to(device)), batch_size=1)
    epochs = 5 # AFP-DE (5)

    for epoch in range(epochs):
        epoch_loss = 0
        for (input, fitness) in dataloader:
            optimizer.zero_grad()
            prediction = model(input)
            loss = loss_function(prediction, fitness)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

        print("Epoch: {} -> mean loss during epoch: {}".format(epoch+1, epoch_loss/(len(dataloader))))
