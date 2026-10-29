from typing import Literal

from open_rigs.core.rig import Rig
from open_rigs.devices import SerialDevice
from open_rigs.vision import Screen, SyncQuad
from pydantic import Field

from open_rigs_hf_visual import __semver__


class MatrixArduino(SerialDevice):
    """Arduino streaming encoder, lick, sync and photodiode values."""


class MatrixArduinoPhotodiode(SerialDevice):
    """Arduino streaming photodiode and sync values."""


class OpenRigsHfVisualRig(Rig):
    version: Literal[__semver__] = __semver__  # type: ignore
    screen: Screen = Field(description="The main display for visual stimuli")
    sync_quad: SyncQuad = Field(description="Synchronisation quad drawn on the display")
    arduino: MatrixArduino = Field(description="Matrix Arduino serial device")
    arduino_photodiode: MatrixArduinoPhotodiode = Field(description="Photodiode Arduino serial device")
