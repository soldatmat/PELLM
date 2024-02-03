from lib.model.extended.model_states import load_model_states
from lib.model.single_layer.SingleLayer import SingleLayer


def init_model(activation_function, state_dict_path: str = None, absolute_path=False):
    model = SingleLayer(activation_function=activation_function)

    # Load learnt weights
    if state_dict_path:
        load_model_states(
            model,
            state_dict_path=state_dict_path,
            absolute_path=absolute_path,
        )

    return model
