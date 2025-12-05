# Import core types
from typing import Literal
from pydantic import Field

from swc.aeon.io import reader
from swc.aeon.schema import BaseSchema, data_reader

from ucl_open_hf_visual import __semver__

# TODO - should inherit from some TaskParameters base class rather than BaseSchema
class UclOpenHfVisualTaskParameters(BaseSchema):
    ...


class UclOpenHfVisualTaskLogic(BaseSchema):
    version: Literal[__semver__] = __semver__
    name: Literal["UclOpenHfVisual"] = Field(default="UclOpenHfVisual", description="Name of the task logic", frozen=True)
    task_parameters: UclOpenHfVisualTaskParameters = Field(description="Parameters of the task logic")
    ...