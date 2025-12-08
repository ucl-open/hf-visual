# Import core types
from typing import Literal, Optional
from pydantic import Field

from swc.aeon.io import reader
from swc.aeon.schema import BaseSchema, data_reader

from ucl_open_hf_visual import __semver__

# TODO - this should be a general class that lives in ucl-open
class TaskParameters(BaseSchema):
    rng_seed: Optional[float] = Field(default=None, description="Seed of the random number generator")

class UclOpenHfVisualTaskParameters(TaskParameters):
    ...


class UclOpenHfVisualTaskLogic(BaseSchema):
    version: Literal[__semver__] = __semver__ # type: ignore
    name: Literal["UclOpenHfVisual"] = Field(default="UclOpenHfVisual", description="Name of the task logic", frozen=True)
    task_parameters: UclOpenHfVisualTaskParameters = Field(description="Parameters of the task logic")
    ...