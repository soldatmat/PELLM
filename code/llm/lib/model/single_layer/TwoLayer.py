import torch


class TwoLayer(torch.nn.Module):
    def __init__(self, activation_function, embedding_size):
        super().__init__()
        self.input_layer = torch.nn.Linear(
            in_features=embedding_size, out_features=embedding_size, bias=False
        )
        self.fitness_head = torch.nn.Linear(
            in_features=embedding_size, out_features=1, bias=True
        )
        self.activation_function = activation_function

    def forward(self, inputs):
        y = self.input_layer(inputs)
        y = self.fitness_head(y)
        y = self.activation_function(y)
        return y
