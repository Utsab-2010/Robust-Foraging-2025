import os
path = "D:/miniconda/envs/mouse/Lib/site-packages/mlagents/trainers/torch/encoders.py"
print(os.path.exists(path))  # Should print True if the file exists, regardless of slashes
print("test")