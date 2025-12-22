import os
import pickle
import mgzip
import torch
import numpy as np
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
from collections import deque
from ood_method.utils import update_cache, compute_cache_logits,get_measures, get_entropy, softmax_entropy
def get_features(model, data_loader, args,device):
    save_path = os.path.join(args.cache_dir, args.model, args.in_dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logits_file_path = os.path.join(save_path, "logits.pkl")
    features_file_path = os.path.join(save_path, "features.pkl")
    if os.path.exists(logits_file_path) and os.path.exists(features_file_path):
        with mgzip.open(logits_file_path, "rb",thread=args.num_workers) as f:
            save_logits = pickle.load(f)
        with mgzip.open(features_file_path, "rb",thread=args.num_workers) as f:
            save_features = pickle.load(f)
    else:

        model.eval()
        features = [[] for i in range(args.num_classes)]
        logits = [[] for i in range(args.num_classes)]
        with torch.no_grad():
            print(data_loader)
            for (images, labels) in tqdm(data_loader):
                images, labels = images.to(device), labels.to(device)
                output, feature = model.get_feature(images)
                
                p_labels = output.argmax(1)
                for i in range(labels.size(0)):
                    logits[p_labels[i]].append(output[i].cpu())
                    features[p_labels[i]].append(feature[i].cpu())

        save_features = []
        save_logits = []
        for i in range(args.num_classes):
            if len(logits[i])==0:
                save_features.append(torch.Tensor([]))
                save_logits.append(torch.Tensor([]))
                continue
            tmp = torch.stack(features[i], dim=0)
            save_features.append(tmp)
            tmp = torch.stack(logits[i], dim=0)
            save_logits.append(tmp)
        with mgzip.open(logits_file_path, "wb",thread=8) as f:
            pickle.dump(save_logits, f)
        with mgzip.open(features_file_path, "wb",thread=8) as f:
            pickle.dump(save_features, f)

    return save_features, save_logits
class ReAct:

    def __init__(self, model,device):
        self.model = model
        self.device = device
        self.linear = model.fc
        '''
        Special Parameters:
            T--Temperature
            p--Truncation Percentage
        '''
        self.T = 1
        self.p = 90

    def set_state(self, threshold):
        self.threshold = threshold

    @torch.no_grad()
    def get_threshold(self, features):
        train_all_features = []
        for i in range(len(features)):
            train_all_features.extend(features[i].cpu().numpy())
        threshold = np.percentile(train_all_features, self.p)
        print(threshold)
        return threshold

    @torch.no_grad()
    def eval(self, args, data_loader):
        self.model.eval()
        result = []
        entropy = np.array([])
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            _, feature = self.model.get_feature(images)
            feature = feature.clip(max=self.threshold)
            output = self.linear(feature)
            loss = softmax_entropy(output)
            prop_entropy = get_entropy(loss, args.num_classes).cpu().numpy()
            entropy = np.concatenate((entropy, prop_entropy))
            output = self.T * torch.logsumexp(output / self.T, dim=1).data.cpu().numpy()
            result.append(output)

        return np.concatenate(result), entropy
    @torch.no_grad()
    def eval_fc(self, args, ood_cfg, data_loader, id_entropy):
        self.model.eval()
        id_conf = np.array([])
        ood_conf = np.array([])
        ood_cache = {}
        tpr = 0.95
        # data_record = []
        #Unpack all hyperparameters
        ood_enabled = ood_cfg['enabled']
        if ood_cfg:
            ood_params = {k: ood_cfg[k] for k in ['shot_capacity', 'alpha', 'lower', 'topk']}
        threshold = np.percentile(id_entropy,ood_params['lower'])
        for images, tags in tqdm(data_loader):
            # sample_record = {}
            images = images.to(self.device)
            _, feature_nonorm = self.model.get_feature(images)
            feature_nonorm = feature_nonorm.clip(max=self.threshold)
            output = self.linear(feature_nonorm)
            feature_norm = feature_nonorm / feature_nonorm.norm(dim=1, keepdim=True)
            loss = softmax_entropy(output)
            prob_map = output.softmax(1)
            pred = torch.argmax(output,dim=1)
            prop_entropy = get_entropy(loss, args.num_classes)

            final_logits = output.clone()
            for i in range(prob_map.shape[0]):
                if ood_enabled and threshold < prop_entropy[i].item():
                    update_cache(ood_cache, int(pred[i].item()), [feature_norm[i].unsqueeze(0), loss[i].item(), prob_map[i].unsqueeze(0),tags[i]], ood_params['shot_capacity'], True)  
                
            if ood_enabled and ood_cache:
                fuzzy_logits = compute_cache_logits(feature_norm, ood_cache, ood_params['alpha'], ood_params['topk'])
                final_logits -= fuzzy_logits
            final_logits = self.T * torch.logsumexp(final_logits / self.T, dim=1).data.cpu().numpy()
            for i in range(len(tags)):
                if tags[i] == 'ID':
                    id_conf = np.concatenate((id_conf, np.array([final_logits[i]])))
                else:
                    ood_conf = np.concatenate((ood_conf, np.array([final_logits[i]])))
        # print("Average_ID_score:", id_conf.mean(axis=0))
        # print("Average_OOD_score:", ood_conf.mean(axis=0))
        # torch.save(data_record,f'/home/enroll2024/yanqi/Storage/lora_tune/checkpoints/ReAct/{ood_name}.pth')
        auroc, aupr, fpr = get_measures(id_conf, ood_conf, tpr)
        return auroc, aupr, fpr

    





