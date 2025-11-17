## LSF job
    bsub -J "scgpt" -W 24:00 -q jupyter_gpu -gpu "num=1:mode=exclusive_process:mig=1" -Is bash

## 수정 사항
- 실행시간 단축을 위해 epochs 15에서 5로 줄여서 실행  

#### key error
    
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

#### training loop 실행 시 OOM

    OutOfMemoryError: CUDA out of memory. Tried to allocate 192.00 MiB. GPU
  
batch_size를 64에서 16으로 줄임   
test loop에서도 동일한 문제가 발생하여 batch_size를 1로 줄임  

#### np.float 에러

    AttributeError: module 'numpy' has no attribute 'float'. np.float was a deprecated alias for the builtin float. To avoid this error in existing code, use float by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use np.float64 here. The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations

astype(np.float) 모두 astype(float)로 변경  
