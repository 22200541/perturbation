## LSF job
    bsub -J "scgpt" -W 24:00 -q jupyter_gpu -gpu "num=1:mode=exclusive_process:mig=2" -Is bash

## Data download
#### Download pretrained scGPT model
    aws s3 sync --no-sign-request s3://czi-scgenept-public/models/pretrained/scgpt models/pretrained/

#### Download pre-computed gene Embeddings
    aws s3 cp --no-sign-request s3://czi-scgenept-public/models/gene_embeddings/GO_C_gene_embeddings-gpt3.5-ada-concat.pickle models/gene_embeddings/

## Training
    python train.py --model-type=scgenept_go_c_gpt --num-epochs=20 --dataset=norman --device=cuda:0
