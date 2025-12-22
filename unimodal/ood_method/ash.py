import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import operator
from utils import *
from ood_method.utils import update_cache, compute_cache_logits,get_measures, get_entropy, softmax_entropy
class ASH:

    def __init__(self, model,device):
        self.model = model
        self.device = device
        self.linear = model.fc

        '''
        Special Parameters:
            T--Temperature
            p--Pruning Percentage
        '''
        self.T = 1
        self.p = 90

    @ torch.no_grad()
    def eval(self, args, data_loader):
        self.model.eval()
        result = []
        entropy = np.array([])
        for (images, _) in tqdm(data_loader):
            images = images.to(self.device)
            _, feature = self.model.get_feature(images)
            output = ash_s_2d(feature, self.p)
            output = self.linear(output)
            loss = softmax_entropy(output)
            prop_entropy = get_entropy(loss, args.num_classes).cpu().numpy()
            output = self.T * torch.logsumexp(output / self.T, dim=1).data.cpu().numpy()
            entropy = np.concatenate((entropy, prop_entropy))
            result.append(output)

        return np.concatenate(result), entropy
    
    @ torch.no_grad()
    def eval_fc(self, args, ood_cfg, data_loader, id_entropy):
        self.model.eval()
        id_conf = np.array([])
        ood_conf = np.array([])
        ood_cache = {}
        tpr = 0.95
        #Unpack all hyperparameters
        ood_enabled = ood_cfg['enabled']
        if ood_cfg:
            ood_params = {k: ood_cfg[k] for k in ['shot_capacity', 'alpha', 'lower', 'topk']}
        threshold = np.percentile(id_entropy, ood_params['lower'])
        for images, tags in tqdm(data_loader):
            images = images.to(self.device)
            _, features = self.model.get_feature(images)
            output = ash_s_2d(features, self.p)
            output = self.linear(output)
            # output = output
            loss = softmax_entropy(output)
            prob_map = output.softmax(1)
            pred = torch.argmax(output,dim=1)
            prop_entropy = get_entropy(loss, args.num_classes)
            final_logits = output.clone()
            for i in range(prob_map.shape[0]):
                if ood_enabled and threshold < prop_entropy[i].item():
                    update_cache(ood_cache, int(pred[i].item()), [features[i].unsqueeze(0), loss[i].item(), prob_map[i].unsqueeze(0),tags[i]], ood_params['shot_capacity'], True)  
            if ood_enabled and ood_cache:
                fuzzy_logits = compute_cache_logits(features, ood_cache, ood_params['alpha'], ood_params['topk'])
                final_logits -= fuzzy_logits
            final_logits = self.T * torch.logsumexp(final_logits / self.T, dim=1).data.cpu().numpy()
            for i in range(len(tags)):
                if tags[i] == 'ID':
                    id_conf = np.concatenate((id_conf, np.array([final_logits[i]])))         
                else:
                    ood_conf = np.concatenate((ood_conf, np.array([final_logits[i]])))
        auroc, aupr, fpr = get_measures(id_conf, ood_conf, tpr)
        return auroc, aupr, fpr




def ash_b(x, percentile):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    return x


def ash_p(x, percentile):
    assert x.dim() == 4
    assert 0 <= percentile <= 100

    b, c, h, w = x.shape

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    return x


def ash_s(x, percentile):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    s2 = x.sum(dim=[1, 2, 3])

    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x

def ash_s_2d(x, percentile=90):
    assert x.dim() == 2
    assert 0 <= percentile <= 100
    b, c = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=1)
    n = x.shape[1]
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c))
    v, i = torch.topk(t, k, dim=1)
    # print(k)
    # print(t.shape)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=1)

    # apply sharpening 
    scale = s1 / s2
    x = x * torch.exp(scale[:, None])
    # print(x.shape)
    return x

def ish(data, percentile):
    x = data.clone()
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    s2 = x.sum(dim=[1, 2, 3])

    scale = s1 / s2
    x = data * torch.exp(scale[:, None, None, None])

    return x