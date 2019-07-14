from typing import NamedTuple
from types import FunctionType

class BaseModelParam(NamedTuple):
    module_name: str
    class_name: str
    input_size: tuple
    layers_trainable: bool
    preprocessing_func: FunctionType