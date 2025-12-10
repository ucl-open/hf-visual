from pathlib import Path
from typing import Union

import pydantic
from ucl_open.rigs.experiment import Experiment
from aind_behavior_services.utils import BonsaiSgenSerializers, convert_pydantic_to_bonsai

import ucl_open_hf_visual.rig
import ucl_open_hf_visual.task

SCHEMA_ROOT = Path("./src/DataSchemas/")
EXTENSIONS_ROOT = Path("./src/Extensions/")
NAMESPACE_PREFIX = "UclOpenHfVisualDataSchema"


def main():
    models = [
        ucl_open_hf_visual.task.UclOpenHfVisualTaskLogic,
        ucl_open_hf_visual.rig.UclOpenHfVisualRig,
        Experiment,
    ]
    model = pydantic.RootModel[Union[tuple(models)]]

    convert_pydantic_to_bonsai(
        model, # type: ignore
        model_name="ucl_open_hf_visual",
        root_element="Root",
        cs_namespace=NAMESPACE_PREFIX,
        json_schema_output_dir=SCHEMA_ROOT,
        cs_output_dir=EXTENSIONS_ROOT,
        cs_serializer=[BonsaiSgenSerializers.JSON],
    )


if __name__ == "__main__":
    main()