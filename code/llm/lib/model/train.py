from math import ceil

import torch

LEARNING_RATE = 1e-3
BATCH_SIZE = 1
N_EPOCHS = 1


def train(
    model: torch.nn.Module,
    train_data,
    train_labels,
    test_data,
    test_labels,
    loss_function,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    n_epochs=N_EPOCHS,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    train_batch_borders = get_batch_borders(len(train_data), batch_size)
    train_n_batches = len(train_batch_borders) - 1
    test_batch_borders = get_batch_borders(len(test_data), batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = init_loss_history(n_epochs, train_n_batches)
    loss_history[0, -1] = evaluation_step(
        model, device, test_data, test_labels, test_batch_borders, loss_function
    )
    for epoch in range(n_epochs):
        for batch in range(train_n_batches):
            training_step(
                model,
                train_data[train_batch_borders[batch] : train_batch_borders[batch + 1]],
                train_labels[
                    train_batch_borders[batch] : train_batch_borders[batch + 1]
                ],
                optimizer,
                loss_function,
            )
            loss_history[epoch + 1, batch] = evaluation_step(
                model,
                device,
                test_data,
                test_labels,
                test_batch_borders,
                loss_function,
                epoch,
                batch,
            )
    return loss_history


def get_batch_borders(n_data, batch_size):
    n_batches = ceil(n_data / batch_size)
    batch_borders = [batch * batch_size for batch in range(n_batches)]
    batch_borders.append(n_data)
    return batch_borders


def init_loss_history(n_epochs, n_batches):
    loss_history = torch.empty((n_epochs + 1, n_batches), dtype=torch.float32)
    loss_history[0, :-1] = float("nan")
    return loss_history


def training_step(model, train_data, train_labels, optimizer, loss_function):
    model.train()

    optimizer.zero_grad()
    output = model(train_data)[:, -1, 0]

    loss = loss_function(output, train_labels)
    loss.backward()
    optimizer.step()


def evaluation_step(
    model,
    device,
    test_data,
    test_labels,
    test_batch_borders,
    loss_function,
    epoch_idx=None,
    batch_idx=None,
):
    model.eval()
    with torch.no_grad():
        outputs = torch.empty(len(test_data)).to(device)
        for batch in range(len(test_batch_borders) - 1):
            outputs[test_batch_borders[batch] : test_batch_borders[batch + 1]] = model(
                test_data[test_batch_borders[batch] : test_batch_borders[batch + 1]]
            )[:, -1, 0]

        loss = loss_function(outputs, test_labels)
        evaluation_print(loss, epoch_idx, batch_idx)
        return loss


def evaluation_print(loss, epoch, batch):
    if epoch is None:
        epoch_str = ""
    else:
        epoch_str = f"Epoch {epoch+1}"
    if epoch is None:
        epoch_sep = ""
    elif batch is None:
        epoch_sep = ": "
    else:
        epoch_sep = ", "
    if batch is None:
        batch_str = ""
    else:
        batch_str = f"batch {batch+1}"
    if batch is None:
        batch_sep = ""
    else:
        batch_sep = ": "
    print(
        epoch_str + epoch_sep + batch_str + batch_sep + f"mean test-data loss = {loss}"
    )
