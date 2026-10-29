import os

from open_rigs.vision import Screen, SyncQuad

from open_rigs_hf_visual.rig import MatrixArduino, MatrixArduinoPhotodiode, OpenRigsHfVisualRig

rig = OpenRigsHfVisualRig(
    root_path="C:/Users/neurogears/source/repos/ucl-open/hf-visual/temp_data",
    screen=Screen(
        window_width=640,
        window_height=480,
        texture_assets_directory="../textures",
    ),
    sync_quad=SyncQuad(
        extent_x=0.1,
        extent_y=0.1,
        location_x=1,
        location_y=1,
    ),
    arduino=MatrixArduino(
        port_name="COM10",
        baud_rate=1000000,
        new_line="\n",
    ),
    arduino_photodiode=MatrixArduinoPhotodiode(
        port_name="COM10",
        baud_rate=250000,
        new_line="\n",
    ),
)


def main(path_seed: str = "./local/{schema}.json"):
    os.makedirs(os.path.dirname(path_seed), exist_ok=True)
    models = [rig]

    for model in models:
        with open(path_seed.format(schema=model.__class__.__name__), "w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2, by_alias=True))


if __name__ == "__main__":
    main()
