# hf-visual
Repository for head-fixed visual sequence stimulation task

## Prerequisites
Windows prerequisites can be installed with 

```
\scripts\windows-requirements.cmd
```

## Deployment
Python and Bonsai environments can be bootsrapped by running

```
\scripts\deploy.cmd
```

To select the appropriate Python environment in VSCode: `Ctrl+Shift+P` >> `Python: Select Interpreter` >> `.\.venv\Scripts\python.exe`

## Generating settings file for an experiment
hf-visual experiments are defined by a set of 3 settings files:

- `UclOpenSession.json`: general metadata
- `UclOpenHfVisualRig.json`: devices and hardware descriptions for the rig
- `UclOpenHfVisualTaskLogic.json`: task-logic specific settings

Example Pythons scripts for generating these settings files are given in the `examples` folder, these should be run with the correct Python environment selected as described in the deployment section above.
After launching the main Bonsai workflow, the paths to these files should be set as a property of the workflow.
