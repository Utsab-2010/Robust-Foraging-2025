# Windows Training Guide

Welcome to **Mouse vs. AI: Robust Visual Foraging Challenge @ NeurIPS 2025**

This is a training guide for **Windows**. For other operating systems, please check:
[Linux](https://github.com/robustforaging/mouse_vs_ai_linux?tab=readme-ov-file#linux-training-guide) and [MacOS](https://github.com/robustforaging/mouse_vs_ai_macOS?tab=readme-ov-file#macos-training-guide)

# Install conda
Open command prompt
```bash
curl -o Miniconda3-latest-Windows-x86_64.exe
start /wait "" Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /AddToPath=1 /RegisterPython=1 /S /D=%USERPROFILE%\Miniconda3
```
To activate conda, do: ```%USERPROFILE%\Miniconda3\Scripts\activate```

# Create conda environment
Open command prompt and navigate to the directory where you want to download the project.

Clone the repository from GitHub:
```bash
git clone https://github.com/robustforaging/mouse_vs_ai_windows.git
cd mouse_vs_ai_windows
```

Then, create and activate the conda environment:
```bash
conda env create -n mouse -f mouse.yml
conda activate mouse
``` 


# Modify file path
Open ```train.py``` and go to line 134 (where ```replace.replace_nature_visual_encoder``` is called).
Update the path to point to the location of ```encoders.py``` in your conda environment.

💡 Tip: The ```encoders.py``` file is usually located in your conda environment’s working directory. For example: ```C:/…/miniconda3/env/mouse2/Lib/site-packages/mlagents/trainers/torch/encoders.py```


# Run script


## Training
```text
Usage: python train.py [options]

Training options:
  --runs-per-network R    Number of runs per network (default: 5)
  --env ID                Run identifier (default: Normal) [defines type of environment]
  --network N1,N2,N3     Comma-separated list of networks to train
                         (default choices: ['fully_connected', 'nature_cnn', 'simple', 'resnet'])
```

Example command for training:
```bash
python train.py --env RandomTrain --network neurips,simple,fully_connected
```
- Troubleshooting: If training only proceeds after pressing ```ENTER```, try running the command with unbuffered output mode:  ```python -u start.py train --runs-per-network 1 --env Normal --network neurips,simple,fully_connected``` 
- If the issue persists, stop the current training episode and train again

## Evaluating
```text
Usage: python evaluate.py [options]

Evaluation options:
  --model      Path to the trained ONNX model file
  --episodes   Number of episodes to run in inference(default: 50)
  --env        Build folder name under ./Builds/
  --log-name   Base name for the output log file
```

Example command for evaluation:
```bash
python evaluate.py --model="./example_model.onnx" --episodes=200
```

# Customize the model
- To change architecture: Add your model to the `/mouse_vs_ai_windows/Encoders` folder
- To change hyperparamters: edit information in `/mouse_vs_ai_windows/Encoders/nature.yaml` file

Then run the above python training script.
