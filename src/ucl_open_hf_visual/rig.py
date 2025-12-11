# Import core types
from typing import Literal
from pydantic import Field

from ucl_open.rigs.base import BaseSchema
from ucl_open.rigs.device import Screen # TODO - change to ucl rigs
from ucl_open.rigs.device import SerialDeviceModule

from ucl_open_hf_visual import __semver__

# TODO - should be part of main package?
# TODO - should be able to define generic number of sync quads (e.g. for different screens)
class SyncQuad(BaseSchema):
    extent_x: float
    extent_y: float
    location_x: float
    location_y: float

class MatrixArduino(SerialDeviceModule):
    device_type: Literal["MatrixArduino"] = "MatrixArduino"

class UclOpenHfVisualRig(BaseSchema):
    version: Literal[__semver__] = __semver__ # type: ignore
    screen: Screen
    sync_quad: SyncQuad
    arduino: MatrixArduino
    ...