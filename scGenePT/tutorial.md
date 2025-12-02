## LSF job
    bsub -J "scgenept" -W 24:00 -q jupyter_gpu -gpu "num=1:mode=exclusive_process:mig=2" -Is bash

## Data download
#### Download pretrained scGPT model
    aws s3 sync --no-sign-request s3://czi-scgenept-public/models/pretrained/scgpt scGenePT/models/pretrained/scgpt

#### Download pre-computed gene Embeddings
    aws s3 cp --no-sign-request s3://czi-scgenept-public/models/gene_embeddings/GO_C_gene_embeddings-gpt3.5-ada-concat.pickle scGenePT/models/gene_embeddings/

#### Download finetuned scGenePT model
    aws s3 sync --no-sign-request s3://czi-scgenept-public/models/finetuned/scgenept_go_c scGenePT/models/finetuned/scgenept_go_c

