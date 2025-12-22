import os
import mgzip
import pickle
import torch
import numpy as np
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
def get_linear_layer_mapping(model_name, dataset, model):
    if model_name == "convnext":
        return model.classifier[-1]
    elif model_name == "convnext_small":
        return model.classifier[-1]
    elif model_name == "convnext_large":
        return model.classifier[-1]
    elif model_name == "vit" and dataset == "ImageNet":
        return model.heads[-1]
    elif model_name == "vit" and dataset in ["cifar10", "cifar100"]:
        return model.head
    elif model_name == "swin" or model_name == "swinv2":
        return model.head
    elif model_name == "densenet" and dataset in ['ImageNet']:
        return model.classifier
    elif model_name == "densenet" and dataset in ['cifar10', 'cifar100']:
        return model.fc
    elif model_name == "regnet":
        return model.fc
    elif model_name == "efficientnet":
        return model.classifier[-1]
    elif model_name == "efficientnet_b7":
        return model.classifier[-1]
    elif model_name == "resnet" and dataset in ['cifar10', 'cifar100']:
        return model.linear
    elif model_name == "resnet" and dataset == "ImageNet":
        return model.fc
    elif model_name == "vgg" and dataset in ['cifar10', 'cifar100']:
        return model.fc
    elif model_name == "mobilenet" and dataset in ['cifar10', 'cifar100']:
        return model.linear
    elif model_name == "maxvit":
        return model.classifier[-1]
    else:
        raise ValueError(f"Unsupported model: {model_name} for dataset: {dataset}")


class OptFS:

    def __init__(self, model, args,device):
        self.model = model
        self.device = device
        self.linear = get_linear_layer_mapping(args.model, args.in_dataset, model)
        self.theta = None
        self.left_boundary = None
        self.width = None
    

    def set_state(self, theta, left_boundary, width):
        self.theta = theta
        self.left_boundary = left_boundary
        self.width = width
    
    @torch.no_grad()
    def get_optimal_shaping(self, features, logits):
        features = torch.cat(features)
        logits = torch.cat(logits)

        preds = torch.softmax(logits, dim=1)
        features = features.cpu().numpy()
        preds = preds.argmax(dim=1).cpu().numpy()


        w = self.linear.weight.data
        b = self.linear.bias.data if self.linear.bias is not None else torch.zeros(w.size(0))
        w = w.cpu().numpy()
        b = b.cpu().numpy()

        left_b = np.quantile(features, 1e-3)
        right_b = np.quantile(features, 1-1e-3)
        
        width = (right_b - left_b) / 100.0
        left_boundary = np.arange(left_b, right_b, width)
        
        lc = w[preds] * features
        lc_fv_list = []
        for b in tqdm(left_boundary):
            mask = (features >= b) & (features < b + width)
            feat_masked = mask * lc
            res = np.mean(np.sum(feat_masked, axis=1))
            lc_fv_list.append(res)
        lc_fv_list = np.array(lc_fv_list)
        theta = lc_fv_list / np.linalg.norm(lc_fv_list, 2) * 1000

        theta = torch.from_numpy(theta[np.newaxis, :])
        return theta, left_boundary, width
    
    @torch.no_grad()
    def eval(self, data_loader):
        self.model.eval()
        result = []

        with torch.no_grad():
            for (images, _) in tqdm(data_loader):
                images = images.to(self.device)
                _, feature = self.model.get_feature(images)
                feature = feature.view(feature.size(0), -1)

                feat_p = torch.zeros_like(feature).to(self.device)
                for i, x in enumerate(self.left_boundary):
                    mask = (feature >= x) & (feature < x + self.width)
                    feat_p += mask * feature * self.theta[0][i]

                output = self.linear(feat_p)
                output /= 200
                loss = softmax_entropy(output)
                prop_entropy = get_entropy(loss, output.shape[1])
                
                result.append(prop_entropy.cpu().numpy())

        return np.concatenate(result)
    
    def run_test(self, data_loader, id_entropy, args, cfg):
        ood_cache = {}    
        #Unpack all hyperparameters
        ood_enabled = cfg['enabled']
        if cfg:
            ood_params = {k: cfg[k] for k in ['shot_capacity', 'alpha', 'lower', 'topk']}
        lower_threshold = np.percentile(id_entropy, ood_params['lower'])
        # print(lower_threshold)
        id_conf = np.array([])
        ood_conf = np.array([])
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for images, tags in tqdm(data_loader):
                images = images.to(device)
                output, feature = self.model.get_feature(images)
                feature_norm = feature / feature.norm(dim=1, keepdim=True)
                pred = output.argmax(dim=1)
                feature = feature.view(feature.size(0), -1)

                feat_p = torch.zeros_like(feature).to(self.device)
                for i, x in enumerate(self.left_boundary):
                    mask = (feature >= x) & (feature < x + self.width)
                    feat_p += mask * feature * self.theta[0][i]

                output = self.linear(feat_p)
                output /= 200
                loss = softmax_entropy(output)
                prop_entropy = get_entropy(loss, output.shape[1])
                prob_map = output.softmax(1)
                final_logits = output.clone()
                for i in range(prob_map.shape[0]):
                    if ood_enabled and lower_threshold < prop_entropy[i].item():
                        update_cache(ood_cache, int(pred[i].item()), [feature_norm[i].unsqueeze(0), loss[i].item(), prob_map[i].unsqueeze(0),tags[i]], ood_params['shot_capacity'], True)  
                if ood_enabled and ood_cache:
                    fuzzy_logits = compute_cache_logits(feature_norm, ood_cache, ood_params['alpha'], ood_params['topk'])
                    # print(fuzzy_logits)
                    final_logits -= fuzzy_logits
                final_logits = torch.logsumexp(final_logits, dim=1).cpu().numpy()
                for i in range(len(tags)):
                        if tags[i] == 'ID':
                            id_conf = np.concatenate((id_conf, np.array([final_logits[i]])))
                        else:
                            ood_conf = np.concatenate((ood_conf, np.array([final_logits[i]])))
        tpr = 0.95
        auroc, aupr, fpr = get_measures(id_conf, ood_conf, tpr)

        return auroc, aupr, fpr

