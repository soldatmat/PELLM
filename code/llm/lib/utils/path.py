import os


def prepare_path(path: str, absolute_path: bool):
    if absolute_path:
        return path
    else:
        return os.getcwd() + path


def prepare_save_paths(save_path, file_prepend, save_name, save_state_dict, save_info):
    if save_state_dict is None:
        save_state_dict = save_path + "/" + file_prepend + "_" + save_name + ".pt"
    if save_info is None:
        save_info = save_path + "/" + file_prepend + "_" + save_name + "_info.pt"
    return save_state_dict, save_info
