# Setup
Date: 2025-11-23  
## HGU HPC
    conda create -n gears python=3.11 pip
    
    conda activate gears
    
    conda install jupyter
    
    cd GEARS/
    
    pip install .
    
    pip uninstall -y torch

    pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
    
    pip install cell-gears


***
    pip show cell-gears
<pre><code> 
Name: cell-gears
Version: 0.1.2
Summary: GEARS
Home-page: https://github.com/snap-stanford/GEARS
Author: Yusuf Roohani, Kexin Huang, Jure Leskovec
Author-email: 
License: MIT
Location: /home/c22200541/miniconda3/envs/gears/lib/python3.11/site-packages
Requires: dcor, networkx, numpy, pandas, scanpy, scikit-learn, scipy, torch, torch_geometric, tqdm
Required-by: 
</code></pre>
