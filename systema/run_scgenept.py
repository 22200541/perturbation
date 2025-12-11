import os.path

from gears import PertData, GEARS
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr
from src.data import get_pert_data
import argparse
from pathlib import Path
from train_systema import load_dataloader
from utils.data_loading import *
from models.scGenePT import *
from sklearn.decomposition import PCA
from gears.inference import evaluate, compute_metrics, deeper_analysis, non_dropout_analysis

# GEARS installation
# ! pip install torch-geometric
# ! pip install cell-gears

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Norman2019')
parser.add_argument('--seed', default=1, type=int)
parser.add_argument("--data_dir", default="data")
parser.add_argument('--outdir', default='results')
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--hiddendim', default=64, type=int)
parser.add_argument('--batchsize', default=32, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--lr', default=1e-3, type=int)
parser.add_argument("--load_model", action="store_true")
parser.set_defaults(load_model=False)

args = parser.parse_args()

def load_trained_scgenept_model(pert_data, model_type, models_dir, model_location, device, verbose = False):
    embs_to_include = get_embs_to_include(model_type)
    vocab_file = models_dir + 'pretrained/scgpt/vocab.json'
    vocab, gene_ids, dataset_genes, gene2idx = match_genes_to_scgpt_vocab(vocab_file, pert_data, True, SPECIAL_TOKENS)
    ntokens = len(vocab)  # size of vocabulary
    genept_embs, genept_emb_type, genept_emb_dim, found_genes_genept = initialize_genept_embeddings(embs_to_include, dataset_genes, vocab, model_type, models_dir)
    go_embs_to_include, go_emb_type, go_emb_dim, found_genes_go = initialize_go_embeddings(embs_to_include, dataset_genes, vocab, model_type, models_dir)
    
    if device == torch.device('cpu'):
        use_fast_transformer = False
    else: 
        use_fast_transformer = False
        
    model = scGenePT(
        ntoken=ntokens,
        d_model=EMBSIZE,
        nhead=NHEAD,
        d_hid=D_HID,
        nlayers=NLAYERS,
        nlayers_cls=N_LAYERS_CLS,
        n_cls=N_CLS,
        vocab=vocab,
        n_perturbagens=2,
        dropout=0.0,
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        pert_pad_id=PERT_PAD_ID,
        use_fast_transformer=use_fast_transformer,
        embs_to_include = embs_to_include,
        genept_embs = genept_embs, 
        genept_emb_type = genept_emb_type, 
        genept_emb_size = genept_emb_dim,
        go_embs_to_include = go_embs_to_include,
        go_emb_type = go_emb_type,
        go_emb_size = go_emb_dim
    )
    
    pretrained_params = torch.load(model_location, weights_only=True, map_location = device)
    # print(pretrained_params)
    if not use_fast_transformer:
        pretrained_params = {
            k.replace("Wqkv.", "in_proj_"): v for k, v in pretrained_params.items()
        }
    
    model.load_state_dict(pretrained_params)
    
    if verbose:
        print(model)
    model.to(device)
    return model, gene_ids

if __name__ == '__main__':
    pert_data = get_pert_data(dataset=args.dataset,
                              seed=args.seed,
                              data_dir=args.data_dir)
    pert_data.get_dataloader(batch_size=args.batchsize,
                             test_batch_size=args.batchsize)
    train_loader = pert_data.dataloader['train_loader']
    val_loader = pert_data.dataloader['val_loader']
    test_loader = pert_data.dataloader['test_loader']
    model_path = f'{args.outdir}/checkpoints/scgenept_seed{args.seed}_{args.dataset}'
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')
    model_name2model_variation = {'scgenept_ncbi+uniprot_gpt' : 'best_model.pt'}
    model_name = 'scgenept_ncbi+uniprot_gpt'
    pretrained_scgpt_model_dir = './models/'
    # load saved model
    print(f"Now loading a {model_name} model ... ")
    print('=' * 30)
    model_filename = model_name2model_variation[model_name]
    if model_name != 'scgpt':
        model_prefix = ''.join(model_name.split('_gpt')[:-1]) 
    else:
        model_prefix = model_name
    model_location = './outputs/norman/scgenept_ncbi+uniprot_gpt/seed_42/models/best_model.pt' 
    scgenept_model, gene_ids = load_trained_scgenept_model(pert_data, model_name, pretrained_scgpt_model_dir, model_location, device)
    print('Done!\n')                               

    # Split train and test
    test_adata = pert_data.adata[pert_data.adata.obs['split'] == 'test']
    train_adata = pert_data.adata[pert_data.adata.obs['split'] == 'train']

    # Get control mean, non control mean (pert_mean), and non control mean differential
    control_adata = train_adata[train_adata.obs['control'] == 1]
    pert_adata = train_adata[train_adata.obs['control'] == 0]
    control_mean = np.array(control_adata.X.mean(axis=0))[0]
    pert_mean = np.array(pert_adata.X.mean(axis=0))[0]
    delta_pert = pert_mean - control_mean

    gene_names = pert_data.gene_names.to_list()
    gene2idx = pert_data.node_map
    cond2name = dict(pert_data.adata.obs[["condition", "condition_name"]].values)
    gene_raw2id = dict(zip(pert_data.adata.var.index.values, pert_data.adata.var.gene_name.values))

    # Store results
    unique_conds = list(set(test_adata.obs['condition'].astype(str).unique()) - set(['ctrl']))
    post_gt_df = pd.DataFrame(columns=pert_data.adata.var['gene_name'].values)
    post_pred_df = pd.DataFrame(columns=pert_data.adata.var['gene_name'].values)
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

        # Get scGenePT predictions
        pert = condition
        scgenept_pred = scgenept_model.pred_perturb_from_ctrl(control_adata, pert, gene_names, device, gene_ids, amp=True, pool_size=300).squeeze()
        post_gt_df.loc[len(post_gt_df)] = X_post
        post_pred_df.loc[len(post_pred_df)] = scgenept_pred

    index = pd.MultiIndex.from_tuples(list(zip(unique_conds, train_counts)), names=['condition', 'n_train'])
    post_gt_df.index = index
    post_pred_df.index = index
    print(post_pred_df.head())
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    post_gt_df.to_csv(f'{args.outdir}/{args.dataset}_{args.seed}_scgenept_post-gt.csv')
    post_pred_df.to_csv(f'{args.outdir}/{args.dataset}_{args.seed}_scgenept_post-pred.csv')