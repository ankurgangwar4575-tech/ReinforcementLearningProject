# 🍄 Mario Double DQN

An AI agent that learns to play Super Mario Bros from stacked grayscale game frames using Double Deep Q-Learning (Double DQN). 🎮

## 📁 Project files

- `environment.py` — creates the Mario environment and preprocesses observations to four `84 x 84` grayscale frames.
- `dqn.py` — defines the convolutional Q-network. 🧠
- `experience_replay.py` — stores past transitions for randomized learning.
- `agent.py` — trains the online and target networks with Double DQN. 🚀
- `test.py` — runs the saved agent with a visible Mario window. 👀
- `parameters.yaml` — contains all adjustable hyperparameters. ⚙️

## 🛠️ Setup

Use Python 3.13, create a virtual environment, then install the packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirement.txt
```

> 💡 If PowerShell blocks activation, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 🏋️ Train

Start training from the project directory:

```powershell
python agent.py
```

The model checkpoint is written to `runs/checkpoints/mario_ddqn.pt`. Every new `python agent.py` command automatically resumes this checkpoint and trains for the configured `steps_per_run` (50,000 by default).

> ⏳ For a quick pipeline check, use `total_steps: 20000`, `steps_per_run: 20000`, and `learning_starts: 1000`. For meaningful learning, restore `total_steps: 1000000`, `steps_per_run: 50000`, and `learning_starts: 10000`.

### 📊 Monitor training with TensorBoard

```powershell
tensorboard --logdir "$env:LOCALAPPDATA\MarioDoubleDQN\tensorboard"
```

Open the local URL TensorBoard prints in the terminal.

## 🕹️ Test

After at least one checkpoint has been saved, run:

```powershell
python test.py
```

This opens a Mario window. Press `Q` or wait for the episode to end to finish evaluation. ✨
