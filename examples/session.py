import datetime
import os

from swc.aeon.schema import Experiment

# TODO - autofill experiment fields
session = Experiment(
    workflow="main.bonsai",
    commit="",
    repository_url="https://github.com/ucl-open/hf-visual"
)

def main(path_seed: str = "./local/{schema}.json"):
    os.makedirs(os.path.dirname(path_seed), exist_ok=True)
    models = [session]

    for model in models:
        with open(path_seed.format(schema=model.__class__.__name__), "w", encoding="utf-8") as f:
            f.write(model.model_dump_json(indent=2))


if __name__ == "__main__":
    main()