import os

from ucl_open_hf_visual.rig import (
    UclOpenHfVisualRig
)
from ucl_open_hf_visual.rig import (
    MatrixArduino,
    SyncQuad
)
from ucl_open.rigs.device import (
    Screen,
)

rig = UclOpenHfVisualRig(
    screen=Screen(
        texture_assets_directory="../textures"
    ),
    sync_quad=SyncQuad(
        extent_x=0.1,
        extent_y=0.1,
        location_x=1,
        location_y=1
    ),
    arduino=MatrixArduino(
        port_name="COM10",
        baud_rate=1000000,
        new_line="\n"
    )
)

def main(path_seed: str = "./local/{schema}.json"):
    os.makedirs(os.path.dirname(path_seed), exist_ok=True)
    models = [rig]

    for model in models:
        with open(path_seed.format(schema=model.__class__.__name__), "w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2, by_alias=True))


if __name__ == "__main__":
    main()