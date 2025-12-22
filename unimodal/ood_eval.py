import os
import time
import argparse
from utils.utils import fix_random_seed
from models.get_model import get_model
from datasets.get_datasets import get_datasets
import torch
import numpy as np
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def process_args():
    parser = argparse.ArgumentParser(description='DCAC OOD detection',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--root_dir', default="/data/Public/Datasets/", type=str)
    parser.add_argument('--model_dir', default=None, type=str)
    parser.add_argument('--in_dataset', default='ImageNet', type=str)
    parser.add_argument('--cache_dir', default='./caches', type=str)
    parser.add_argument('--result_dir', default='./results', type=str)
    parser.add_argument('--seed', default=3, type=int)
    parser.add_argument('--method', default='msp', type=str,choices=['msp','energy','react','ash','cadref',"optfs","maxlogits"])
    parser.add_argument('--logits_method', default='energy', type=str)
    parser.add_argument('--model', type=str, default='resnet',
                        choices=['resnet','densenet','vit','swin','convnext','regnet','efficientnet','maxvit'])
    parser.add_argument('--test_batch_size', default=512, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
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
    
    if args.in_dataset == "ImageNet":
        args.num_classes = 1000
    elif args.in_dataset == "cifar10":
        args.num_classes = 10
    elif args.in_dataset == "cifar100":
        args.num_classes = 100
    train_loader, ood_loaders, cfg = get_datasets(args)
    model = get_model(args)
    model.to(device)
    model.eval()
    
    if args.method == 'optfs':
        from ood_method.OptFs import OptFS,get_features
        ood_detector = OptFS(model, args, device)
        shaping_parameter_file_path = os.path.join(args.cache_dir, args.model, args.in_dataset, "OptFS_shaping_parameter.pkl")
        if os.path.exists(shaping_parameter_file_path) and args.use_feature_cache:
            with open(shaping_parameter_file_path, "rb") as f:
                theta, left_boundary, width = pickle.load(f)
        else:
            features, logits = get_features(model, train_loader, args, device)
            theta, left_boundary, width = ood_detector.get_optimal_shaping(features, logits)
            with open(shaping_parameter_file_path, "wb") as f:
                pickle.dump((theta, left_boundary, width), f)
        
        ood_detector.set_state(theta, left_boundary, width)
    
    elif args.method =='cadref':
        from ood_method.CADRef import CADRef,get_features
        ood_detector = CADRef(model, device)
        cadref_file_path = os.path.join(args.cache_dir, args.model, args.in_dataset, "CADRef_"+args.logits_method+".pkl")
        if os.path.exists(cadref_file_path) and args.use_feature_cache:
            with open(cadref_file_path, "rb") as f:
                train_mean, global_mean_logit_score = pickle.load(f)
        else:
            features, logits = get_features(model, train_loader, args, device)
            train_mean, global_mean_logit_score = ood_detector.get_state(features, logits)
            with open(cadref_file_path, "wb") as f:
                pickle.dump((train_mean, global_mean_logit_score), f)
        
        ood_detector.set_state(train_mean, global_mean_logit_score)

    elif args.method == 'react':
        from ood_method.ReAct import ReAct,get_features
        ood_detector = ReAct(model, device)
        features, logits = get_features(model, train_loader, args, device)
        threshold = ood_detector.get_threshold(features)
        ood_detector.set_state(threshold)
    elif args.method == 'ash':
        from ood_method.ash import ASH
        ood_detector = ASH(model, device)

    if os.path.exists(os.path.join(args.cache_dir, args.model, args.in_dataset, args.method, 'train_id_entropy.npy')):
        id_entropy = np.load(os.path.join(args.cache_dir, args.model, args.in_dataset, args.method, 'train_id_entropy.npy'))
        # print(id_entropy.shape)
    else:
        if args.method in ['msp', 'energy', 'maxlogits','cadref']:
            from ood_method.msp import get_train_entropy
            id_entropy = get_train_entropy(args, train_loader, device, model)
            np.save(os.path.join(args.cache_dir, args.model, args.in_dataset, args.method, 'train_id_entropy.npy'), id_entropy)
        elif args.method == 'optfs':
            id_entropy = ood_detector.eval(train_loader)
            np.save(os.path.join(args.cache_dir, args.model, args.in_dataset, args.method, 'train_id_entropy.npy'), id_entropy)
        elif args.method == 'react':
            _, id_entropy = ood_detector.eval(args,train_loader)
            np.save(os.path.join(args.cache_dir, args.model, args.in_dataset, args.method, 'train_id_entropy.npy'), id_entropy)
        elif args.method == 'ash':
            _, id_entropy = ood_detector.eval(args, train_loader)
            np.save(os.path.join(args.cache_dir, args.model, args.in_dataset, args.method, 'train_id_entropy.npy'), id_entropy)
    print("ID train entropy computed.")
    auroc_list = []
    fpr_list = []
    if args.method in ['msp', 'energy', 'maxlogits']:
        from ood_method.msp import run_test
        for dataset_name, dataloader in ood_loaders.items():
            auroc, aupr, fpr = run_test(args, dataloader, model, cfg, id_entropy)
            auroc_list.append(auroc)
            fpr_list.append(fpr)
            print(f"OOD Dataset: {dataset_name}, AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, FPR95: {fpr:.4f}")
            with open(os.path.join(args.result_dir, args.model,args.in_dataset, args.method,"results.txt"), "a") as f:
                f.write("{:20} {:10} {:10} {:10}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), dataset_name, auroc, fpr))
    
    elif args.method == 'optfs':
        for dataset_name, dataloader in ood_loaders.items():
            auroc, aupr, fpr = ood_detector.run_test(dataloader, id_entropy, args, cfg)
            auroc_list.append(auroc)
            fpr_list.append(fpr)
            print(f"OOD Dataset: {dataset_name}, AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, FPR95: {fpr:.4f}")
            with open(os.path.join(args.result_dir, args.model,args.in_dataset, args.method,"results.txt"), "a") as f:
                f.write("{:20} {:10} {:10} {:10}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), dataset_name, auroc, fpr))
    elif args.method == 'cadref':
        for dataset_name, dataloader in ood_loaders.items():
            auroc, aupr, fpr = ood_detector.eval_fc(args, cfg, dataloader, id_entropy)
            auroc_list.append(auroc)
            fpr_list.append(fpr)
            print(f"OOD Dataset: {dataset_name}, AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, FPR95: {fpr:.4f}")
            with open(os.path.join(args.result_dir, args.model,args.in_dataset, args.method,"results.txt"), "a") as f:
                f.write("{:20} {:10} {:10} {:10}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), dataset_name, auroc, fpr))
    elif args.method == 'react':
        for dataset_name, dataloader in ood_loaders.items():
            auroc, aupr, fpr = ood_detector.eval_fc(args, cfg, dataloader, id_entropy)
            auroc_list.append(auroc)
            fpr_list.append(fpr)
            print(f"OOD Dataset: {dataset_name}, AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, FPR95: {fpr:.4f}")
            with open(os.path.join(args.result_dir, args.model,args.in_dataset, args.method,"results.txt"), "a") as f:
                f.write("{:20} {:10} {:10} {:10}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), dataset_name, auroc, fpr))
    elif args.method == 'ash':
        for dataset_name, dataloader in ood_loaders.items():
            auroc, aupr, fpr = ood_detector.eval_fc(args, cfg, dataloader, id_entropy)
            auroc_list.append(auroc)
            fpr_list.append(fpr)
            print(f"OOD Dataset: {dataset_name}, AUROC: {auroc:.4f}, AUPR: {aupr:.4f}, FPR95: {fpr:.4f}")
            with open(os.path.join(args.result_dir, args.model,args.in_dataset, args.method,"results.txt"), "a") as f:
                f.write("{:20} {:10} {:10} {:10}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), dataset_name, auroc, fpr))
    print(f'Mean AUROC: {np.mean(auroc_list)*100:.2f}%, Mean FPR(0.95): {np.mean(fpr_list)*100:.2f}%')
    with open(os.path.join(args.result_dir, args.model,args.in_dataset, args.method,"results.txt"), "a") as f:
        f.write("{:20} {:10} {:10} {:10}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 'Average', np.mean(auroc_list), np.mean(fpr_list)))
    

    
if __name__ == '__main__':
    main()