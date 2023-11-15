import os


def prepare_path(path: str, absolute_path: bool):
    if absolute_path:
        return path
    else:
        return os.getcwd() + path
