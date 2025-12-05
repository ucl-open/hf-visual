# Import core types
from typing import Literal

from swc.aeon.io import reader
from swc.aeon.schema import BaseSchema, data_reader

from ucl_open_hf_visual import __semver__


class UclOpenHfVisualRig(BaseSchema):
    version: Literal[__semver__] = __semver__
    ...