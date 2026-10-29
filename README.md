# hf-visual
Repository for head-fixed visual sequence stimulation task, built on the [open-rigs](https://github.com/open-rigs/open-rigs) framework.

## Prerequisites
Windows prerequisites can be installed with 

```
\scripts\windows-requirements.cmd
```

## Deployment
Python and Bonsai environments can be bootstrapped by running

```
\scripts\deploy.cmd
```

This will:

1. Create the Python environment with `uv` and install the `open_rigs_hf_visual` package together with the shared `open_rigs` schemas.
2. Restore the `Bonsai.Sgen` dotnet tool and regenerate the JSON schema and Bonsai extension classes (see [Experiment schemas](#experiment-schemas)).
3. Create the Bonsai environment in `.bonsai` and install the required packages.

To select the appropriate Python environment in VSCode: `Ctrl+Shift+P` >> `Python: Select Interpreter` >> `.\.venv\Scripts\python.exe`

## Bonsai packages
The main workflow uses the `OpenRigs.Core`, `OpenRigs.Devices` and `OpenRigs.Logging` Bonsai packages from open-rigs. These are not published on NuGet.org yet, so prebuilt packages (version `42.42.42-dev0`, built from open-rigs commit `77094a4`) are checked in under `local_packages`, which is configured as a package source in `.bonsai/NuGet.config`.

To rebuild them from a newer open-rigs commit:

```
git clone https://github.com/open-rigs/open-rigs
cd open-rigs
dotnet pack src/OpenRigs.Core/OpenRigs.Core.csproj -c Release
dotnet pack src/OpenRigs.Devices/OpenRigs.Devices.csproj -c Release
dotnet pack src/OpenRigs.Logging/OpenRigs.Logging.csproj -c Release
```

then copy the `.nupkg` files from `artifacts/package/release` into `local_packages` and update the package versions in `.bonsai/Bonsai.config` if they changed.

## Experiment schemas
Experiment parameters are defined as pydantic models in `src/open_rigs_hf_visual`:

- `rig.py`: rig hardware (screen, sync quad, Arduino serial devices), extending the shared `open_rigs` device definitions
- `task.py`: task logic parameters
- `data_types.py`: data records streamed by the rig Arduinos

After changing any of these, regenerate the combined JSON schema and the Bonsai extension classes with

```
uv run regenerate-schemas
```

This writes `src/DataSchemas/open_rigs_hf_visual.json` and `src/Extensions/OpenRigsHfVisual.Generated.cs`.

## Generating settings file for an experiment
hf-visual experiments are defined by a set of 3 settings files:

- `ExperimentSession.json`: session metadata (subject id, session id, repository info)
- `OpenRigsHfVisualRig.json`: devices and hardware descriptions for the rig, including the root path for data logging
- `OpenRigsHfVisualTaskLogic.json`: task-logic specific settings

Example Python scripts for generating these settings files are given in the `examples` folder, these should be run with the correct Python environment selected as described in the deployment section above.
After launching the main Bonsai workflow, the paths to these files should be set as a property of the workflow.
