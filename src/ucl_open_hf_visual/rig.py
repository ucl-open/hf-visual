# Import core types
from typing import Literal
from pydantic import Field

from ucl_open.rigs.base import BaseSchema
from ucl_open_hf_visual.visual_stimulation import Screen # TODO - change to ucl rigs

from ucl_open_hf_visual import __semver__


class UclOpenHfVisualRig(BaseSchema):
    version: Literal[__semver__] = __semver__ # type: ignore
    screen: Screen
    ...