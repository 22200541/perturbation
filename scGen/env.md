# Setup
Date: 2025-11-11  
## HGU HPC
    conda create -n scgen python=3.11 pip
    
    conda activate scgen
    
    conda install jupyter
    
    cd scgen-master/
    
    pip install .
    
    pip install scgen

GEARS에서 Norman data를 사용하기 위한 패키지

    pip install cell-gears

    pip install torch_geometric

systema 사용을 위한 패키지

    pip install gseapy
    
    pip install plottable
***
    pip show scgen
<pre><code> 
Name: scgen
Version: 2.1.1
Summary: ScGen - Predicting single cell perturbations.
Home-page: https://github.com/theislab/scgen
Author: Mohammad lotfollahi
Author-email: mohammad.lotfollahi@helmholtz-muenchen.de
License: MIT
Location: /home/c22200541/miniconda3/envs/scgen/lib/python3.11/site-packages
Requires: adjustText, anndata, matplotlib, scanpy, scvi-tools, seaborn
Required-by:  
</code></pre>

torch: 2.9.0+cu128  
numpy: 2.3.4  
scanpy: 1.11.5  
scvi-tools: 1.4.0.post1
