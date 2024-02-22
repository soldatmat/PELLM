import torch


class SingleLayer(torch.nn.Module):
    def __init__(self, activation_function, embedding_size):
        super().__init__()
        self.fitness_head = torch.nn.Linear(in_features=embedding_size, out_features=1, bias=True)
        self.activation_function = activation_function

    def forward(self, x):
        y = self.activation_function(self.fitness_head(x))
        return y
