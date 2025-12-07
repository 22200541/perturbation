## Data download
#### Download pretrained scGPT model
    aws s3 sync --no-sign-request s3://czi-scgenept-public/models/pretrained/scgpt scGenePT/models/pretrained/scgpt

#### Download pre-computed gene Embeddings
    aws s3 cp --no-sign-request s3://czi-scgenept-public/models/gene_embeddings/NCBI+UniProt_embeddings-gpt3.5-ada.pkl scGenePT/models/gene_embeddings/

#### Download finetuned scGenePT model
    aws s3 sync --no-sign-request s3://czi-scgenept-public/models/finetuned/scgenept_go_c scGenePT/models/finetuned/scgenept_go_c

## Training
### LSF job
    bsub -J "scgenept" -Is -W 24:00 -q normal -gpu "num=1:mode=exclusive_process:mig=2" -n 4 -R "rusage[mem=10000]" bash

### Run
batch size 16으로 낮춤

    python train.py --model-type=scgenept_ncbi+uniprot_gpt --num-epochs=20 --dataset=norman --device=cuda:0 \
        --batch-size 16 --eval-batch-size 16

## Inference
### LSF job
    bsub -J "scgenept" -W 24:00 -q jupyter_gpu -gpu "num=1:mode=exclusive_process:mig=2" -Is bash

