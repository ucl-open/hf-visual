# Running the analysis notebooks in VS Code

A step-by-step guide to running `psth.ipynb` (stimulus-triggered running speed and PSTHs) on a fresh Windows
machine. If a step is already done on your machine, skip it.

## Part 1: Install the tools (once per computer)

Open **PowerShell** (Start menu → type "PowerShell" → open it) and run these one at a time.

1. **Git** (downloads code from GitHub):

   ```
   winget install --id Git.Git -e
   ```

2. **uv** (sets up Python and all the packages, including the right Python version):

   ```
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **VS Code**:

   ```
   winget install --id Microsoft.VisualStudioCode -e
   ```

Close PowerShell and open a new one so it picks up the new tools. Check that they work:

```
git --version
uv --version
```

Each should print a version number. If you see "not recognized", restart the computer and try again.

4. **VS Code extensions.** Open VS Code, click the Extensions icon in the left sidebar (four squares), then search
   for and install **Python** (by Microsoft) and **Jupyter** (by Microsoft).

## Part 2: Get the code (once)

In PowerShell, go to the folder where you keep code, for example:

```
cd ~\Documents\GitHub
```

Download the repository (sign in to GitHub if a window pops up), go into it and switch to the analysis branch:

```
git clone https://github.com/ucl-open/hf-visual.git
cd hf-visual
git switch open-rigs-analysis-code
```

## Part 3: Set up Python (once, and again whenever the code changes)

In the `hf-visual` folder, run:

```
uv sync
```

This creates a `.venv` folder containing Python 3.11 and every package the notebooks need. The first time takes a
few minutes.

## Part 4: Open the notebook in VS Code

1. **File → Open Folder…** and choose the `hf-visual` folder. If asked "Do you trust the authors?", click **Yes**.
2. In the file list, open **examples → psth.ipynb**.
3. **Pick the right Python.** Click **Select Kernel** in the top-right corner of the notebook, choose
   **Python Environments…**, and pick **.venv** (Python 3.11), whose path ends in `hf-visual\.venv\Scripts\python.exe`.
   If VS Code offers to install `ipykernel`, you picked the wrong Python: go back and choose `.venv`.

## Part 5: Point it at your data

1. Make sure the **Z: drive** is connected: in File Explorer you should be able to browse to
   `Z:\ibn-vision\DATA\SUBJECTS`. If not, connect to the network or VPN first.
2. In the cell under the **Session** heading, set the paths for the session you want:

   ```python
   root = Path(r"Z:\ibn-vision\DATA\SUBJECTS\M26031\stimuli\20261008\M26031_20261008_1\ses-20261008_1_date-2026-10-08T15-12-29")
   ephys_root = Path(r"Z:\ibn-vision\DATA\SUBJECTS\M26031\ephys\20261008\20261008_1\M26031_20261008_1_g2")
   kilosort_dir = None
   ```

   - `root` is the **stimuli** session folder, the one containing `Arduino`, `Texture` and so on.
   - `ephys_root` is the **SpikeGLX** folder, the one ending in `_g0`, `_g1`, `_g2` …
   - `kilosort_dir` is the spike sorting. It can be the day's CatGT **Processed** folder (e.g.
     `...\ephys61008\Processed_1`): the spikes of the `_gN` run in `ephys_root` are picked out and put on its
     clock automatically. Leave it `None` when there is no sorting for that run. On 2026-10-08 the probe data of
     `_g2` (the texture session) is missing from `Processed_1`, so keep `None` there.
   - Keep the `r` in front of each path's quotes; it stops Windows backslashes from causing errors.

## Part 6: Run it

Click **Run All** at the top of the notebook, or press **Shift+Enter** to run one cell at a time.

| Section | What you should see |
|---|---|
| Session | Takes about 10–30 s over the network, then prints `Arduino -> NI-DAQ: ClockMap(piecewise, …)` |
| Sync check | A smooth wavy line: the Arduino clock drifting (normal) |
| Stimulus presentations | `N of N onsets matched to a photodiode edge` and a table of trials |
| Alignment check | The photodiode jumps up sharply at time 0 in the left plot |
| Running speed | Speed over the session, the stimulus-triggered average, a trial-by-trial heatmap, per-texture bars |
| Spikes | Stops with **"No spike sorting loaded"**. That is expected until `kilosort_dir` is set |

Settings worth changing:

- `cm_per_count` (running-speed cell): wheel circumference divided by encoder counts per revolution, for speed in
  cm/s instead of encoder counts/s.
- `running_threshold`: the speed above which a trial counts as running.
- `window`: the time range around each stimulus.
- `photodiode_channel`: the NI-DAQ channel with the screen photodiode (`XA3` on this rig).

## Other runs: flat CSV files (`psth_oldBonsai.ipynb`)

Older workflows (`DotMotion_SpeedTuning`, `DriftingGratings_Direction`, ...) save each stream as its own CSV file in
one folder, for example `M26031_DriftingGratings_Direction_WheelLog2026-10-08T16_00_40.csv`. For these, open
**examples → psth_oldBonsai.ipynb** instead and set, in its first cells:

- `folder`: the folder holding the CSV files (e.g. `...\stimuli61008\M26031_20261008_1`).
- `protocol`: part of the protocol name (e.g. `"DriftingGratings"`), and `start` if it was run more than once.
- `kilosort_dir`: the day's CatGT **Processed** folder (`...\ephys61008\Processed_1`), or `None`.
- `ephys_root`: the SpikeGLX `_gN` folder recorded **during that run**. The `fileCreateTime` line in its
  `.nidq.meta` file is its start time; on 2026-10-08, `_g0` is DotMotion (15:36), `_g1` is DriftingGratings (16:00)
  and `_g2` is the texture session.

If the wrong `_gN` folder is given, loading stops with "does not seem to be the recording of this run".

With spike sorting, the notebook ends with a table of units ranked by how strongly their firing differs between
conditions, rasters and PSTHs of the most selective units, and a population heatmap.

## Exploring units across runs (`explore_units.ipynb`)

Open **examples → explore_units.ipynb**, set `day` in the **Settings** cell and click **Run All**. Loading takes
about a minute per run. Each day lists its runs, one per SpikeGLX gate, with the kind of stimulus:

| Day | `_g0` | `_g1` | `_g2` | `_g3` | `_g4` |
|---|---|---|---|---|---|
| 2026-10-06 (`Processed_1`) | dot motion | gratings, spatial frequency | (no `_g2`) | gratings, temporal frequency | |
| 2026-10-07 (`Processed_1`) | gratings, direction | gratings, spatial frequency | textures | | |
| 2026-10-08 (`Processed_1`) | dot motion | gratings, direction | textures (no spikes: probe meta file missing) | | |
| 2026-10-09 (`Processed_1`) | gratings, spatial frequency | gratings, temporal frequency | textures | | |
| 2026-10-13 (`Processed_1`) | dot motion | gratings, temporal frequency | gratings, spatial frequency | sparse noise | checkerboard |
| 2026-10-14 (`Processed_0`) | sparse noise | checkerboard | gratings, direction | textures | gratings, temporal frequency |
| 2026-10-18 (`Processed_0`) | gratings, direction | checkerboard | sparse noise | textures & noise | |

Spatial and temporal frequency runs are shown at two directions, so their conditions read e.g. `sf0.020 180`.

To add a day, copy one of the entries in `days` and set its folders, `kilosort_dir` and the `kind` of each run:
`trials` (gratings, dot motion), `reversals` (checkerboard), `sparse_noise` or `textures` (open-rigs `ses-...`
sessions). The day's `Processed_N\..._gkey.csv` lists which stimulus was run in each gate.

At the bottom there are controls to:

- pick the **recording** used to filter and sort units, and the unit **label** (good / mua / all);
- set a **minimum selectivity** (how much a unit's firing differs between stimulus conditions; about 1 = no effect);
- **ignore blank** trials, to rank units by their tuning among stimuli (direction, speed) rather than by
  stimulus-versus-blank;
- **sort** by selectivity, response, depth or unit number, and step through units with **previous / next**.

Each unit is shown in every run side by side (raster, PSTH, tuning curve; for sparse noise, its ON/OFF receptive
field maps and response time course) with its position on the probe. If the
controls don't appear, run `uv sync` again and reload VS Code (they need the `ipywidgets` package).

## When the code is updated

```
git pull
uv sync
```

Then click **Restart** at the top of the notebook and **Run All** again.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` (e.g. matplotlib) | Wrong kernel. Redo Part 4, step 3, and pick `.venv` |
| `.venv` is not in the kernel list | Run `uv sync` again, then in VS Code: **Ctrl+Shift+P → Developer: Reload Window** |
| `FileNotFoundError` or "no Arduino CSV file found" | The `root` path is wrong, or the Z: drive is not connected |
| "gate N has no probe data" | The probe files of that `_gN` run were missing when it was sorted; set `kilosort_dir = None` |
| "no NI-DAQ stream found" | `ephys_root` must be the `_gN` folder containing the `.nidq.bin` file |
| "does not seem to be the recording of this run" | Wrong `_gN` folder for this run: compare `fileCreateTime` in the `.nidq.meta` files with the run's start time |
| "no consistent pulse offset found" | The stimulus and ephys folders are from different recordings: check dates and run numbers |
| Few onsets matched to the photodiode | Try another `photodiode_channel` (`XA2`, `XA4`, `XA5`) |
| Very slow | Normal over the network for the first cells. Keep `align_probe=False` and `check_probe_range = False` |
