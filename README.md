# Windows Training Guide

Welcome to **Mouse vs. AI: Robust Visual Foraging Challenge @ NeurIPS 2025**

This is a training guide for **Windows**. For other operating systems, please check:
[Linux](https://github.com/robustforaging/mouse_vs_ai_linux) and [macOS](https://github.com/robustforaging/mouse_vs_ai_macOS?tab=readme-ov-file#macos-training-guide)

# Install conda
Open command prompt
```bash
curl -o Miniconda3-latest-Windows-x86_64.exe
start /wait "" Miniconda3-latest-Windows-x86_64.exe /InstallationType=JustMe /AddToPath=1 /RegisterPython=1 /S /D=%USERPROFILE%\Miniconda3
```
To activate conda, do: ```%USERPROFILE%\Miniconda3\Scripts\activate```

# Create conda environment
```cd``` into your working directory

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
```bash
python start.py train --runs-per-network 1 --env Normal --network neurips,simple,fully_connected
```
- Troubleshooting: If training only proceeds after pressing ```ENTER```, try running the command with unbuffered output mode:  ```python -u start.py train --runs-per-network 1 --env Normal --network neurips,simple,fully_connected``` 
- If the issue persists, stop the current training episode and train again

## Evaluating
TODO





