import torch

from lib.utils.path import prepare_path


def save_pt_file(var, save_to: str, absolute_path=False, var_name: str = None):
    file_path = prepare_path(save_to, absolute_path)
    var_name = f'{var_name} ' if var_name else ""
    print(f'saving {var_name}to file "{file_path}"')
    torch.save(var, file_path)
