import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import clip
import argparse
from utils.utils import fix_random_seed
import torch
import numpy as np
from models.get_model import get_model
from datasets.get_datasets import get_datasets
from methods.coop_test import run_test_coop
from methods.local_prompt_test import run_test_local_prompt
import time
def process_args():
    parser = argparse.ArgumentParser(description='DCAC OOD detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_dir', default="/data/Public/Datasets/", type=str)
    parser.add_argument('--model_dir', default=None, type=str)
    parser.add_argument('--in_dataset', default='ImageNet', type=str)
    parser.add_argument('--cache_dir', default='./caches', type=str)
    parser.add_argument('--result_dir', default='./results', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--method', default='mcm', type=str,choices=['mcm','locoop', 'sct', 'ospcoop','coop','local-prompt'])
    parser.add_argument('--model', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14','RN50'])
    parser.add_argument('--test_batch_size', default=512, type=int)
    args = parser.parse_args()
    return args

def main():
    args = process_args()
    fix_random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for dir_path in [
        args.cache_dir,
        args.result_dir,
        os.path.join(args.cache_dir, args.model),
        os.path.join(args.result_dir, args.model),
        os.path.join(args.cache_dir, args.model, args.in_dataset),
        os.path.join(args.result_dir, args.model, args.in_dataset),
        os.path.join(args.cache_dir, args.model, args.in_dataset,args.method),
        os.path.join(args.result_dir, args.model, args.in_dataset,args.method)
    ]:
        os.makedirs(dir_path, exist_ok=True)

    _, preprocess = clip.load(args.model, device='cpu')
    train_loader, ood_loaders, classnames, cfg = get_datasets(args,preprocess=preprocess)
    args.num_classes = len(classnames)
    args.classnames = classnames
    model, clip_weights = get_model(args,device)
    model.eval()
    if os.path.exists(os.path.join(args.cache_dir, args.model, args.in_dataset, args.method, 'train_id_entropy.npy')):
        id_entropy = np.load(os.path.join(args.cache_dir, args.model, args.in_dataset, args.method, 'train_id_entropy.npy'))
    else:
        id_entropy = model.get_train_id_entropy(train_loader, device)
        np.save(os.path.join(args.cache_dir, args.model, args.in_dataset, args.method, 'train_id_entropy.npy'), id_entropy)
    print("ID train entropy computed.")
    auroc_list = []
    fpr_list = []
    if args.method in ['mcm','locoop', 'sct', 'ospcoop','coop']:
        for dataset_name, dataloader in ood_loaders.items():
            auroc, aupr, fpr = run_test_coop(cfg, dataloader, model, clip_weights,id_entropy)
            auroc_list.append(auroc)
            fpr_list.append(fpr)
            print(f'OOD dataset: {dataset_name}, AUROC: {auroc*100:.2f}%, AUPR: {aupr*100:.2f}%, FPR(0.95): {fpr*100:.2f}%')
            with open(os.path.join(args.result_dir, args.model,args.in_dataset, args.method,"results.txt"), "a") as f:
                f.write("{:20} {:10} {:10} {:10}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), dataset_name, auroc, fpr))
    elif args.method == 'local-prompt':
        for dataset_name, dataloader in ood_loaders.items():
            auroc, aupr, fpr = run_test_local_prompt(cfg, dataloader, model,id_entropy)
            auroc_list.append(auroc)
            fpr_list.append(fpr)
            print(f'OOD dataset: {dataset_name}, AUROC: {auroc*100:.2f}%, AUPR: {aupr*100:.2f}%, FPR(0.95): {fpr*100:.2f}%')
            with open(os.path.join(args.result_dir, args.model,args.in_dataset, args.method,"results.txt"), "a") as f:
                f.write("{:20} {:10} {:10} {:10}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), dataset_name, auroc, fpr))      
    print(f'Mean AUROC: {np.mean(auroc_list)*100:.2f}%, Mean FPR(0.95): {np.mean(fpr_list)*100:.2f}%')
    with open(os.path.join(args.result_dir, args.model,args.in_dataset, args.method,"results.txt"), "a") as f:
        f.write("{:20} {:10} {:10} {:10}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Average', np.mean(auroc_list), np.mean(fpr_list)))

main()