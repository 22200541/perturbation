# Setup
2025-11-11  
## HGU HPC
    conda create -n scgpt python=3.11 pip
    
    conda activate scgpt
    
    conda install jupyter
    
    cd scGPT/
    
    pip install .
    
    pip install scgpt
    
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 "torch==2.3.0" "torchvision==0.18.0" "torchaudio==2.3.0"
    
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

### Tutorial

    bsub -J "scgpt" -W 24:00 -q jupyter_gpu -gpu "num=1:mode=exclusive_process:mig=1" -Is bash

- 실행시간 단축을 위해 epochs 15에서 5로 줄여서 실행  

##### key error
    
    RuntimeError: Error(s) in loading state_dict for TransformerGenerator: Unexpected key(s) in state_dict: "transformer_encoder.layers.0.self_attn.Wqkv.weight", "transformer_encoder.layers.0.self_attn.Wqkv.bias",

수정 전 코드

    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if any([k.startswith(prefix) for prefix in load_param_prefixs])
    }

수정 후 코드

    pretrained_dict = {
        k: v
        for k, v in pretrained_raw.items()
        if any([k.startswith(prefix) for prefix in load_param_prefixs])
           and k in model_dict
           and v.shape == model_dict[k].shape
    }

##### training loop 실행 시 OOM

    OutOfMemoryError: CUDA out of memory. Tried to allocate 192.00 MiB. GPU
  
batch_size 64에서 16으로 줄임  

##### np.float 에러

    AttributeError: module 'numpy' has no attribute 'float'. np.float was a deprecated alias for the builtin float. To avoid this error in existing code, use float by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use np.float64 here. The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations

astype(np.float) 모두 astype(float)로 변경  
