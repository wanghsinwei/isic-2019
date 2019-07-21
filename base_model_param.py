from typing import NamedTuple
from types import FunctionType

BaseModelParam = NamedTuple('BaseModelParam', [
    ('module_name', str),
    ('class_name', str),
    ('input_size', tuple),
    ('preprocessing_func', FunctionType)
])