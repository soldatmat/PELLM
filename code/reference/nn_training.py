# An example of nn training code
# author: Vojtěch Brejtr

import torch
import torch.nn as nn
from src.models.net import get_resnet50
from src.datasets.datasets import get_dataloader
from utils.metrics import AUC

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(dataset_train: str, dataset_validation: str, model_weights: str = None) -> None:


    batch_size = 5
    learning_rate = 1e-3
    n_epochs = 100

    model = get_resnet50(model_weights)
    model = model.to(DEVICE)

    loss_fcn = nn.functional.binary_cross_entropy_with_logits
    optimizer = torch.optim.Adam(model.parameters(), learning_rate=learning_rate)

    trainloader = get_dataloader(dataset_train, train=True, batch_size=batch_size)
    validloader = get_dataloader(dataset_validation, train=False, batch_size=1)


    for epoch in range(n_epochs):
        model.train()
        for it, blob in enumerate(trainloader):
            images, labels = [x.to(DEVICE) for x in blob]
            
            optimizer.zero_grad()

            predictions = model(images)

            loss = loss_fcn(predictions, labels)
            loss.backward()

            optimizer.step()

            # Per-iteration info print
            if it % 25 == 0:
                with torch.no_grad():
                    print(f'\rTRAINING: EPOCH: {epoch + 1} ITERATION: [{it} / {len(trainloader)}] LOSS: {str(round(loss.item(), 3)).zfill(3)}', end='')

        # Per-epoch validation
        print(f"\n EVAL: EPOCH {epoch + 1}: ")
        model.eval()
        with torch.no_grad():
            predictions = torch.zeros(len(validloader))
            labels = torch.zeros(len(validloader))

            for i, blob in enumerate(validloader):
                img, label = [x.to(DEVICE) for x in blob]
                predictions[i] = model(img)
                labels[i] = label
            
            auc, roc = AUC(predictions, label)

if __name__ == "__main__":
    train()