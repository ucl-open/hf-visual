import os

from ucl_open_hf_visual.rig import (
    UclOpenHfVisualRig
)
from ucl_open.rigs.device import (
    SerialDeviceModule
)
from ucl_open.rigs.device import (
    Screen,
)

rig = UclOpenHfVisualRig(
    screen=Screen(
        texture_assets_directory="../textures"
    ),
    arduino=SerialDeviceModule(
        port_name="COM10",
        baud_rate=1000000,
        new_line="\n",
        pattern="%d"
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