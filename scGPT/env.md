# Setup
Date: 2025-11-11  
## HGU HPC
    conda create -n scgpt python=3.11 pip
    
    conda activate scgpt
    
    conda install jupyter
    
    cd scGPT/
    
    pip install .
    
    pip install scgpt

torch와 torchtext 버전 충돌로 torch 버전 낮춤
    
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 "torch==2.3.0" "torchvision==0.18.0" "torchaudio==2.3.0"

torchtext 설치 시 torch가 최신 버전으로 업데이트 되어 --no-deps 적용
    
    pip install --no-cache-dir --no-deps "torchtext==0.18.0"
    
    pip install torch_geometric

***
    pip show scgpt
<pre><code> 
Name: scGPT
Version: 0.2.4
Summary: Large-scale generative pretrain of single cell using transformer.
Home-page: https://github.com/bowang-lab/scGPT
Author: Haotian
Author-email: subercui@gmail.com
License: MIT
Location: /home/c22200541/miniconda3/envs/scgpt/lib/python3.11/site-packages
Requires: cell-gears, datasets, leidenalg, numba, orbax, pandas, scanpy, scib, scikit-misc, scvi-tools, torch, torchtext, typing-extensions, umap-learn
Required-by: 
</code></pre>

