import os
import pickle
import mgzip
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
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
class CADRef:
    def __init__(self, model,device):
        self.model = model
        self.device = device
        self.logit_method = 'Energy'
        self.linear = model.fc
        self.train_mean = None
        self.global_mean_logit_score = None
    def set_state(self, train_mean, global_mean_logit_score):
        self.train_mean = train_mean
        self.global_mean_logit_score = global_mean_logit_score

    @torch.no_grad()
    def get_mean_feature(self, features):
        train_mean = []
        for i in range(len(features)):
            train_mean.append(features[i].mean(dim=0))
        train_mean = torch.stack(train_mean)
        return train_mean
    
    @torch.no_grad()
    def get_global_mean_logit_score(self,logits):
        train_logit_score = []
        for i in range(len(logits)):
            train_logit_score.append(self.get_logits_score(logits[i]))
        mean_logit_score = torch.mean(torch.cat(train_logit_score),dim=0)
        return mean_logit_score
    
    @torch.no_grad()
    def get_state(self,features,logits):
        train_mean = self.get_mean_feature(features)
        global_mean_logit_score = self.get_global_mean_logit_score(logits)
        return train_mean, global_mean_logit_score
        
    def get_logits_score(self,logits):
        if self.logit_method == "MaxLogit":
            return maxLogits(logits)
        elif self.logit_method == "GEN":
            return gen(logits)
        elif self.logit_method == "Energy":
            return energy(logits)
        elif self.logit_method == "MSP":
            return msp(logits)

    @torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        result = []
        w = self.linear.weight.data
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            logit, feature = self.model.get_feature(images)
            class_ids = torch.argmax(torch.softmax(logit, dim=1), dim=1).cpu().numpy()
            tm = self.train_mean[class_ids].to(self.device)
            dist = feature - tm
            sg = w[class_ids].sign()
            ep_dist = dist * sg
            ep_dist[ep_dist<0] = 0
            ep_error = ep_dist.norm(dim=1,p=1)/feature.norm(dim=1,p=1)

            en_dist = dist*(-sg)
            en_dist[en_dist<0] = 0
            en_error = en_dist.norm(dim=1,p=1)/feature.norm(dim=1,p=1)

            logit_score = self.get_logits_score(logit)  
            # print(logit_score)
            score = ep_error/logit_score + en_error/self.global_mean_logit_score

            result.append(-score.cpu().numpy())
        return np.concatenate(result)

    @torch.no_grad()
    def eval_fc(self, args, ood_cfg, data_loader, id_entropy):
        id_conf = np.array([])
        ood_conf = np.array([])
        self.model.eval()
        w = self.linear.weight.data
        ood_cache = {} 
        ood_enabled = ood_cfg['enabled']
        if ood_cfg:
            ood_params = {k: ood_cfg[k] for k in ['shot_capacity', 'alpha', 'lower', 'topk']}
        lower_threshold = np.percentile(id_entropy, ood_cfg['lower'])
        for images, tags in tqdm(data_loader):
            images = images.to(self.device)
            logit, feature = self.model.get_feature(images)
            loss = softmax_entropy(logit)
            prob_map = logit.softmax(1)
            pred = torch.argmax(logit,dim=1)
            prop_entropy = get_entropy(loss, args.num_classes)
            feature_norm = feature / feature.norm(dim=-1,keepdim=True)
            class_ids = torch.argmax(torch.softmax(logit, dim=1), dim=1).cpu().numpy()
            tm = self.train_mean[class_ids].to(self.device)
            dist = feature - tm
            sg = w[class_ids].sign()
            ep_dist = dist * sg
            ep_dist[ep_dist<0] = 0
            ep_error = ep_dist.norm(dim=1,p=1)/feature.norm(dim=1,p=1)

            en_dist = dist*(-sg)
            en_dist[en_dist<0] = 0
            en_error = en_dist.norm(dim=1,p=1)/feature.norm(dim=1,p=1)

            final_logits = logit.clone()
                # print(final_logits)
            for i in range(prob_map.shape[0]):
                if ood_enabled and lower_threshold < prop_entropy[i].item():
                    update_cache(ood_cache, int(pred[i].item()), [feature_norm[i].unsqueeze(0), loss[i].item(), prob_map[i].unsqueeze(0),tags[i]], ood_params['shot_capacity'], True)  
            if ood_enabled and ood_cache:
                fuzzy_logits = compute_cache_logits(feature_norm, ood_cache, ood_params['alpha'], ood_params['topk'])
                final_logits -= fuzzy_logits
            logit_score = self.get_logits_score(final_logits)
            # print(logit_score)
            score = (ep_error/logit_score + en_error/self.global_mean_logit_score).cpu().numpy()
            for i in range(len(tags)):
                    if tags[i] == 'ID':
                        id_conf = np.concatenate((id_conf, np.array([-score[i]])))
                    else:
                        ood_conf = np.concatenate((ood_conf, np.array([-score[i]])))
        tpr = 0.95
        auroc, aupr, fpr = get_measures(id_conf, ood_conf, tpr)
        return auroc, aupr, fpr







def maxLogits(output):
    scores = output.max(dim=1).values
    return scores

def gen(output,gamma=0.1):
    M = output.shape[-1]//10
    M = 10 if M <10 else M
    smax = (F.softmax(output, dim=1))
    probs_sorted = torch.sort(smax, dim=1).values[:,-M:]
    scores = torch.sum(probs_sorted ** gamma * (1 - probs_sorted) ** gamma, axis=1)
    return 1/scores

def energy(output):
    return torch.logsumexp(output, dim=1)

def msp(output):
    return torch.max(F.softmax(output, dim=1), dim=1).values

