## LSF job
    bsub -J "scgpt" -W 24:00 -q jupyter_gpu -gpu "num=1:mode=exclusive_process:mig=2" -Is bash

## 수정 사항
- 실행시간 단축을 위해 epochs 15에서 10으로 줄여서 실행
- data_name = "norman" 

#### key error
    
    RuntimeError: Error(s) in loading state_dict for TransformerGenerator: Unexpected key(s) in state_dict: "transformer_encoder.layers.0.self_attn.Wqkv.weight", "transformer_encoder.layers.0.self_attn.Wqkv.bias",

수정 전 코드

    load_param_prefixs = [
        "encoder",
        "value_encoder",
        "transformer_encoder",
    ]

수정 후 코드

    load_param_prefixs = None  

#### training loop 실행 시 OOM

    OutOfMemoryError: CUDA out of memory. Tried to allocate 192.00 MiB. GPU
  
batch_size를 64에서 32로 줄임   

#### np.float 에러

    AttributeError: module 'numpy' has no attribute 'float'. np.float was a deprecated alias for the builtin float. To avoid this error in existing code, use float by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use np.float64 here. The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations

astype(np.float) 모두 astype(float)로 변경  
