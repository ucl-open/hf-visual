from aeon.io.api import load
from aeon.io.reader import Reader
from aeon.schema.streams import Device
from aeon.schema import core as stream
from aeon.schema.streams import Stream, StreamGroup
import aeon.io.reader as _reader
from dotmap import DotMap
from pathlib import Path

class Texture(Stream):
    """Texture change log."""
    def __init__(self, pattern):
        super().__init__(_reader.Csv(f"{pattern}_*", ["texture"]))

class Arduino(Stream):
    def __init__(self, pattern):
        super().__init__(_reader.Csv(f"{pattern}_*", ["encoder_count", "lick_count_left", "lick_count_right", "last_sync_pulse_time", "photodiode_value", "current_ms"]))

class ArduinoPhotodiode(Stream):
    def __init__(self, pattern):
        super().__init__(_reader.Csv(f"{pattern}_*", ["photodiode_value", "sync_value"]))
    

exp = DotMap(
    [
        Device("Texture", Texture),
        Device("Arduino", Arduino),
        Device("ArduinoPhotodiode", ArduinoPhotodiode),
        Device("VideoData", stream.Video),
    ]
)

print(exp.VideoData)

data_dir = Path("C:/Users/neurogears/source/repos/ucl-open/hf-visual/temp_data/ses-20251217_1_date-2025-12-17T17-27-21/")

data = load(data_dir, exp.VideoData.Video)
print(data)