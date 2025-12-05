# Import core types
from typing import Literal
from pydantic import Field

from swc.aeon.io import reader
from swc.aeon.schema import BaseSchema, Device, data_reader
from ucl_open_hf_visual.visual_stimulation import Screen

from ucl_open_hf_visual import __semver__


class UclOpenHfVisualRig(BaseSchema):
    version: Literal[__semver__] = __semver__
    screen: Screen
    ...