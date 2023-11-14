from math import ceil

import torch


def train(
    model: torch.nn.Module,
    train_data,
    train_labels,
    test_data,
    test_labels,
    loss_function,
    batch_size=1,
    learning_rate=1e-3,
    n_epochs=1,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    n_batches = ceil(len(train_data) / batch_size)
    batch_borders = [batch * batch_size for batch in range(n_batches)]
    batch_borders.append(len(train_data))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = init_loss_history(n_epochs, n_batches)
    loss_history[0, -1] = evaluation_step(
        model, device, test_data, test_labels, loss_function
    )
    for epoch in range(n_epochs):
        for batch in range(n_batches):
            training_step(
                model,
                train_data[batch_borders[batch] : batch_borders[batch + 1]],
                train_labels[batch_borders[batch] : batch_borders[batch + 1]],
                optimizer,
                loss_function,
            )
            loss_history[epoch + 1, batch] = evaluation_step(
                model, device, test_data, test_labels, loss_function, epoch, batch
            )
    return loss_history


def init_loss_history(n_epochs, n_batches):
    loss_history = torch.empty((n_epochs + 1, n_batches), dtype=torch.float32)
    loss_history[0, :-1] = float("nan")
    return loss_history


def training_step(model, train_data, train_labels, optimizer, loss_function):
    model.train()

    optimizer.zero_grad()
    output = model(train_data).logits[:, -1, 0]

    loss = loss_function(output, train_labels)
    loss.backward()
    optimizer.step()


def evaluation_step(
    model, device, test_data, test_labels, loss_function, epoch=None, batch=None
):
    model.eval()
    with torch.no_grad():
        outputs = torch.zeros(len(test_data)).to(device)
        losses = torch.zeros(len(test_data)).to(device)

        for d in range(len(test_data)):
            input = test_data[d]
            label = test_labels[d]
            outputs[d] = model(input).logits[-1]
            losses[d] = loss_function(outputs[d], label)
        loss = torch.mean(losses)

        evaluation_print(loss, epoch, batch)
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
