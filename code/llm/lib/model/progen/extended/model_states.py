import torch

from lib.utils.path import prepare_path


def load_model_states(model, state_dict_path: str, absolute_path=False):
    file_path = prepare_path(state_dict_path, absolute_path)
    print(f'loading model state dict from file "{file_path}"')
    model.load_state_dict(torch.load(file_path))


def save_model_states(model, save_to: str, absolute_path=False):
    file_path = prepare_path(save_to, absolute_path)
    print(f'saving model state dict to file "{file_path}"')
    torch.save(model.state_dict(), file_path)
