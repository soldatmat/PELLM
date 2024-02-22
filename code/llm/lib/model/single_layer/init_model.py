from lib.model.extended.model_states import load_model_states
from lib.model.single_layer.SingleLayer import SingleLayer


def init_model(model, activation_function, embedding_size, state_dict_path: str = None, absolute_path=False):
    model = model(activation_function, embedding_size)

    # Load learnt weights
    if state_dict_path:
        load_model_states(
            model,
            state_dict_path=state_dict_path,
            absolute_path=absolute_path,
        )

    return model
