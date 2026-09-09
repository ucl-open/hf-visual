import os

from ucl_open_hf_visual.task import (
    UclOpenHfVisualTaskLogic,
    UclOpenHfVisualTaskParameters,
)

task_logic = UclOpenHfVisualTaskLogic(
    task_parameters=UclOpenHfVisualTaskParameters(
        available_textures=["SN_605_1","SN_605_2","SN_605_3","SN_605_4","SN_605_5",
                            "SN_1_1","SN_1_2","SN_1_3","SN_1_4","SN_1_5",
                            "SImg_605_1","SImg_605_2","SImg_605_3","SImg_605_4","SImg_605_5",
                            "SImg_1_1","SImg_1_2","SImg_1_3","SImg_1_4","SImg_1_5",
                            "blank1","blank2"],
        inter_presentation_texture="blank1",
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