import os

from ucl_open_hf_visual.task import (
    UclOpenHfVisualTaskLogic,
    UclOpenHfVisualTaskParameters,
)

task_logic = UclOpenHfVisualTaskLogic(
    task_parameters=UclOpenHfVisualTaskParameters(
        available_textures=["SN_1_1", "SynthIm_1_1", "SynthIm_1_5"],
        inter_presentation_texture="blankStim_1",
        presentation_time=0.25,
        inter_presentation_time=0.75
    ),
)

def main(path_seed: str = "./local/{schema}.json"):
    example_task_logic = task_logic
    os.makedirs(os.path.dirname(path_seed), exist_ok=True)
    models = [example_task_logic]

    for model in models:
        with open(path_seed.format(schema=model.__class__.__name__), "w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2, by_alias=True))


if __name__ == "__main__":
    main()