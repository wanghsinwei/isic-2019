from typing import NamedTuple
from types import FunctionType

class BaseModelParam(NamedTuple):
    input_size: tuple
    base_model: FunctionType
    base_model_name: str
    layers_trainable: bool
    preprocessing_func: FunctionType