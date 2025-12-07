import os.path

from gears import PertData, GEARS
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from data import get_pert_data
import argparse
from pathlib import Path
import torch
import scanpy as sc
import scgen
import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Norman2019')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument("--data_dir", default="data")
parser.add_argument('--outdir', default='results')
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--hiddendim', default=64, type=int)
parser.add_argument('--batchsize', default=32, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--lr', default=1e-3, type=int)
parser.add_argument("--load_model", action="store_true")
parser.set_defaults(load_model=False)
parser.add_argument("--early_stop", default=25, type=int)
parser.add_argument("--perturbed", default="SAMD1+ZBTB1")

args = parser.parse_args()

if __name__ == '__main__':
    pert_data = get_pert_data(dataset=args.dataset,
                              seed=args.seed,
                              data_dir=args.data_dir)
    pert_data.get_dataloader(batch_size=args.batchsize,
                             test_batch_size=args.batchsize)
    pert_adata = pert_data.adata
    
    train_adata = pert_adata[pert_adata.obs['split'] == 'train']
    val_adata = pert_adata[pert_adata.obs['split'] == 'val']
    test_adata = pert_adata[pert_adata.obs['split'] == 'test']
    
    train_new = train_adata.concatenate(val_adata, index_unique=None) # train, val 합치기
    idx_list = []
    for cond, df in test_adata.obs.groupby('condition'):
        # test set에 있는 perturbation에 대해서 각 perturbation 당 하나씩 뽑기
        chosen_idx = np.random.choice(df.index, size=1, replace=False)[0] 
        idx_list.append(chosen_idx)
    subset_adata = test_adata[idx_list].copy()
    
    # train set에 뽑은 데이터 추가
    train_new = train_new.concatenate(subset_adata, index_unique=None)
    train_new = train_new.copy()
    
    model_path = f'{args.outdir}/checkpoints/scgen_seed{args.seed}_{args.dataset}'
    
    scgen.SCGEN.setup_anndata(train_new, batch_key="condition", labels_key="cell_type")
    scgen_model = scgen.SCGEN(train_new)

    if args.load_model and os.path.exists(model_path):
        print(f'Loading model from {model_path}')
        scgen_model.load_pretrained(model_path)
    else:
        scgen_model.train(
            max_epochs=args.epochs,
            batch_size=args.batchsize,
            early_stopping=True,
            early_stopping_patience=args.early_stop
        )
        Path(f'{args.outdir}/checkpoints').mkdir(parents=True, exist_ok=True)
        scgen_model.save(model_path, overwrite=True)

    # Split train and test
    test_adata = pert_adata[pert_adata.obs['split'] == 'test']
    train_adata = pert_adata[pert_adata.obs['split'] == 'train']

    # Get control mean, non control mean (pert_mean), and non control mean differential
    control_adata = train_adata[train_adata.obs['control'] == 1]
    pert_adata = train_adata[train_adata.obs['control'] == 0]
    control_mean = np.array(control_adata.X.mean(axis=0))[0]
    pert_mean = np.array(pert_adata.X.mean(axis=0))[0]
    delta_pert = pert_mean - control_mean

    # Store results
    unique_conds = list(set(test_adata.obs['condition'].astype(str).unique()) - set(['ctrl']))
    post_gt_df = pd.DataFrame(columns=pert_adata.var['gene_name'].values)
    post_pred_df = pd.DataFrame(columns=pert_adata.var['gene_name'].values)
    train_counts = []
    for condition in tqdm(unique_conds):
        gene_list = condition.split('+')
        if 'ctrl' in gene_list:
            gene_list.remove('ctrl')

        # Select adata condition
        adata_condition = test_adata[test_adata.obs['condition'] == condition]
        X_post = np.array(adata_condition.X.mean(axis=0))[
            0]  # adata_condition.X.mean(axis=0) is a np.matrix of shape (1, n_genes)

        # Store number of train perturbations
        n_train = 0
        for g in gene_list:
            if f'{g}+ctrl' in train_adata.obs['condition'].values:
                n_train += 1
            elif f'ctrl+{g}' in train_adata.obs['condition'].values:
                n_train += 1
        train_counts.append(n_train)

        # Get scGen predictions
        #scgen_pred = list(scgen_model.predict([gene_list]).values())[0]
        pred_adata, delta_adata = scgen_model.predict(
            ctrl_key = "ctrl",
            stim_key = args.perturbed,           
            celltype_to_predict = "A549"  
        )
        # 예측된 세포들의 평균 발현 벡터
        scgen_pred = np.array(pred_adata.X.mean(axis=0))[0]
        post_gt_df.loc[len(post_gt_df)] = X_post
        post_pred_df.loc[len(post_pred_df)] = scgen_pred

    index = pd.MultiIndex.from_tuples(list(zip(unique_conds, train_counts)), names=['condition', 'n_train'])
    post_gt_df.index = index
    post_pred_df.index = index

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    post_gt_df.to_csv(f'{args.outdir}/{args.dataset}_{args.seed}_scgen_post-gt.csv')
    post_pred_df.to_csv(f'{args.outdir}/{args.dataset}_{args.seed}_scgen_post-pred.csv')