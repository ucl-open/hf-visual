# Import core types
from typing import List, Literal

from open_rigs.core.task import TaskParameters
from pydantic import Field
from swc.aeon.schema import BaseSchema

from open_rigs_hf_visual import __semver__


class OpenRigsHfVisualTaskParameters(TaskParameters):
    available_textures: List[str] = Field(description="Names of the textures that can be presented")
    inter_presentation_texture: str = Field(description="Name of the texture shown between presentations")
    presentation_time: float = Field(description="Duration (seconds) of each texture presentation")
    inter_presentation_time: float = Field(description="Duration (seconds) between texture presentations")


class OpenRigsHfVisualTaskLogic(BaseSchema):
    version: Literal[__semver__] = __semver__  # type: ignore
    name: Literal["OpenRigsHfVisual"] = Field(default="OpenRigsHfVisual", description="Name of the task logic", frozen=True)
    task_parameters: OpenRigsHfVisualTaskParameters = Field(description="Parameters of the task logic")
