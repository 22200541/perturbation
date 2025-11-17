    ---------------------------------------------------------------------------
    AttributeError                            Traceback (most recent call last)
    Cell In[16], line 1
    ----> 1 latent_X = model.get_latent_representation()
          2 latent_adata = sc.AnnData(X=latent_X, obs=train_new.obs.copy())
    
    File ~/miniconda3/envs/scgen/lib/python3.11/site-packages/torch/utils/_contextlib.py:120, in context_decorator.<locals>.decorate_context(*args, **kwargs)
        117 @functools.wraps(func)
        118 def decorate_context(*args, **kwargs):
        119     with ctx_factory():
    --> 120         return func(*args, **kwargs)
    
    File ~/miniconda3/envs/scgen/lib/python3.11/site-packages/scvi/model/base/_vaemixin.py:346, in VAEMixin.get_latent_representation(self, adata, indices, give_mean, mc_samples, batch_size, return_dist, dataloader, **data_loader_kwargs)
        344     qzm: Tensor = outputs.get(MODULE_KEYS.QZM_KEY)
        345     qzv: Tensor = outputs.get(MODULE_KEYS.QZV_KEY)
    --> 346     qz: Distribution = Normal(qzm, qzv.sqrt())
        348 if return_dist:
        349     qz_means.append(qzm.cpu())
    
    AttributeError: 'NoneType' object has no attribute 'sqrt'
