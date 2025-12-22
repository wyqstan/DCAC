import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ood_method.utils import update_cache, compute_cache_logits,get_measures, get_entropy, softmax_entropy
from tqdm import tqdm
def get_train_entropy(args,train_loader, device, model):
    entropies = np.array([])
    with torch.no_grad():
        for images, _ in tqdm(train_loader):
            images = images.to(device)
            outputs = model(images)
            loss = softmax_entropy(outputs)
            entropy = get_entropy(loss, args.num_classes)
            entropies = np.concatenate((entropies, entropy.cpu().numpy()))
    return entropies

def run_test(args, dataloader, model, cfg, id_entropy):
    ood_cache = {}    
    #Unpack all hyperparameters
    ood_enabled = cfg['enabled']
    if cfg:
        ood_params = {k: cfg[k] for k in ['shot_capacity', 'alpha', 'lower', 'topk']}
    lower_threshold = np.percentile(id_entropy, ood_params['lower'])
    # print(lower_threshold)
    id_conf = np.array([])
    ood_conf = np.array([])
    device = next(model.parameters()).device
    with torch.no_grad():
        for images, tags in tqdm(dataloader):
            images = images.to(device)
            logits, features = model.get_feature(images)
            features = features / features.norm(dim=1, keepdim=True)
            loss = softmax_entropy(logits)
            prob_map = logits.softmax(1)
            pred = torch.argmax(logits,dim=1)
            prop_entropy = get_entropy(loss, args.num_classes)
            # print(prop_entropy)
            final_logits = logits.clone()
            for i in range(prob_map.shape[0]):
                if ood_enabled and lower_threshold < prop_entropy[i].item():
                    update_cache(ood_cache, int(pred[i].item()), [features[i].unsqueeze(0), loss[i].item(), prob_map[i].unsqueeze(0),tags[i]], ood_params['shot_capacity'], True)  
            if ood_enabled and ood_cache:
                fuzzy_logits = compute_cache_logits(features, ood_cache, ood_params['alpha'], ood_params['topk'])
                final_logits -= fuzzy_logits
            if args.method == 'msp':
                final_logits = F.softmax(final_logits,dim=-1).cpu().numpy()
                ood_score = np.max(final_logits,axis=1)
            elif args.method == 'energy':
                T = 1
                ood_score = T * torch.logsumexp(final_logits / T, dim=1).data.cpu().numpy()
            elif args.method == 'maxlogits':
                ood_score = np.max(final_logits,axis=1)
            for i in range(len(tags)):
                if tags[i] == 'ID':
                    id_conf = np.concatenate((id_conf, np.array([ood_score[i]])))
                else:
                    ood_conf = np.concatenate((ood_conf, np.array([ood_score[i]])))
    tpr = 0.95
    auroc, aupr, fpr = get_measures(id_conf, ood_conf, tpr)
    return auroc, aupr, fpr
    