import torch


def train(
    model: torch.nn.Module,
    train_data,
    train_labels,
    test_data,
    test_labels,
    loss_function,
    batch_size=1, # TODO implement batches
    learning_rate=1e-3,
    n_epochs=1,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    train_data = [data.to(device) for data in train_data]
    train_labels = torch.tensor(train_labels).to(device)
    test_data = [data.to(device) for data in test_data]
    test_labels = torch.tensor(test_labels).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    evaluation_step(model, device, test_data, test_labels, loss_function)

    for epoch in range(n_epochs):
        training_step(model, device, train_data, train_labels, optimizer, loss_function)
        evaluation_step(model, device, test_data, test_labels, loss_function)


def training_step(model, device, train_data, train_labels, optimizer, loss_function):
    model.train()
    for d in range(len(train_data)):
        input = train_data[d]
        label = train_labels[d]
        
        optimizer.zero_grad()
        output = model(input).logits[-1]

        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()


def evaluation_step(model, device, test_data, test_labels, loss_function):
    model.eval()
    with torch.no_grad():
        outputs = torch.zeros(len(test_data)).to(device)
        losses = torch.zeros(len(test_data)).to(device)

        for d in range(len(test_data)):
            input = test_data[d]
            label = test_labels[d]
            outputs[d] = model(input).logits[-1]
            losses[d] = loss_function(outputs[d], label)

        # TODO evaluate
        print(f"Training: mean test loss = {torch.mean(losses)}")
