from lib.model.progen.init_model import init_model as init_progen_model
from lib.model.extended.model_states import load_model_states
from lib.model.extended.TransformerExtended import TransformerExtended


def init_model(state_dict_path: str = None, absolute_path=False):
    transformer = init_progen_model()

    # Disable learning for original layers
    for param in transformer.parameters():
        param.requires_grad = False

    model = TransformerExtended(transformer=transformer)

    # Load learned weights
    if state_dict_path:
        load_model_states(
            model,
            state_dict_path=state_dict_path,
            absolute_path=absolute_path,
        )

    return model
