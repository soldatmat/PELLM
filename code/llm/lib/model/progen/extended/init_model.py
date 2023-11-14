import torch

from lib.model.progen.init_model import init_model as init_progen_model
from lib.model.progen.extended.model_states import load_model_states


def init_model(
    device: torch.device = None, state_dict_path: str = None, absolute_path=False
):
    model = init_progen_model(device=device)

    # Disable learning for original layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace lm_head with a new layer
    model.lm_head = torch.nn.Linear(in_features=1024, out_features=1, bias=True).to(
        device
    )

    # Load learned weights
    if state_dict_path:
        load_model_states(
            model, state_dict_path=state_dict_path, absolute_path=absolute_path
        )

    return model
