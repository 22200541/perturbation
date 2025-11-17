#### Setup
  conda create -n scgpt python=3.11 pip
  conda activate scgpt
  conda install jupyter
  cd scGPT/
  pip install .
  pip install scgpt
  pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 "torch==2.3.0" "torchvision==0.18.0" "torchaudio==2.3.0"
  pip install --no-cache-dir --no-deps "torchtext==0.18.0"
  pip install torch_geometric
