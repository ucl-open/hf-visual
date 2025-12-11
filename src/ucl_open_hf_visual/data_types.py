from ucl_open.rigs.base import BaseSchema

class MatrixArduinoData(BaseSchema):
    encoder_count: int
    lick_count_left: int
    lick_count_right: int
    last_sync_pulse_time: int
    photodiode_val: int
    current_ms: int