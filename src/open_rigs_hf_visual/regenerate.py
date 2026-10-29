import json
import os
from pathlib import Path
from typing import Union

import pydantic
from open_rigs.core import ExperimentSession

import open_rigs_hf_visual.data_types
import open_rigs_hf_visual.rig
import open_rigs_hf_visual.task

SCHEMA_ROOT = Path("./src/DataSchemas/")
SCHEMA_FILE = SCHEMA_ROOT / "open_rigs_hf_visual.json"
EXTENSIONS_ROOT = Path("./src/Extensions/")
NAMESPACE = "OpenRigsHfVisual"


def main():
    models = [
        open_rigs_hf_visual.task.OpenRigsHfVisualTaskLogic,
        open_rigs_hf_visual.rig.OpenRigsHfVisualRig,
        ExperimentSession,
        open_rigs_hf_visual.data_types.MatrixArduinoData,
        open_rigs_hf_visual.data_types.MatrixArduinoPhotodiodeData,
    ]
    model = pydantic.RootModel[Union[tuple(models)]]
    schema = model.model_json_schema(by_alias=True, mode="serialization", union_format="primitive_type_array")
    SCHEMA_ROOT.mkdir(parents=True, exist_ok=True)
    SCHEMA_FILE.write_text(json.dumps(schema, indent=2))
    print(f"Schema written to {SCHEMA_FILE}")
    os.system(
        f"dotnet bonsai.sgen {SCHEMA_FILE} --namespace {NAMESPACE} --output {EXTENSIONS_ROOT} "
        "--serializer json --serializer yaml"
    )


if __name__ == "__main__":
    main()
