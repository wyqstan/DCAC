import torch
import math
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from methods.utils import update_cache, compute_cache_logits,get_measures
def get_entropy(loss, clip_weights):
    max_entropy = math.log2(clip_weights.size(0))
    return loss / max_entropy
def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
def get_clip_logits_coop(images, model, clip_weights):
    with torch.no_grad():
        if isinstance(images, list):
            images = torch.cat(images, dim=0).cuda()
        else:
            images = images.cuda()

        image_features, image_features_local = model.image_encoder(images.type(model.dtype))
        image_features /= image_features.norm(dim=-1, keepdim=True)
        clip_logits = 100. * image_features @ clip_weights.T

        if image_features.size(0) > 1:
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = torch.argmax(clip_logits,dim=1)
            # print(pred.shape)
        else:
            loss = softmax_entropy(clip_logits)
            prob_map = clip_logits.softmax(1)
            pred = int(clip_logits.topk(1, 1, True, True)[1].t()[0])

        return image_features, clip_logits, loss, prob_map, pred
def run_test_coop(ood_cfg, loader,model, clip_weights,id_entropy):
    with torch.no_grad():
        ood_cache = {}
        #Unpack all hyperparameters
        ood_enabled = ood_cfg['enabled']
        if ood_cfg:
            ood_params = {k: ood_cfg[k] for k in ['shot_capacity', 'alpha', 'lower',  'topk']}
        threshold = np.percentile(id_entropy, ood_params['lower'])
        id_conf = np.array([])
        ood_conf = np.array([])
        T = 1
        tpr = 0.95
        for images, tags in tqdm(loader):
            images = images.cuda()
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits_coop(images ,model, clip_weights)
            prop_entropy = get_entropy(loss, clip_weights)
            final_logits = clip_logits.clone()
            for i in range(prob_map.shape[0]):
                if ood_enabled and threshold < prop_entropy[i].item():
                    update_cache(ood_cache, int(pred[i].item()), [image_features[i].unsqueeze(0), loss[i].item(), prob_map[i].unsqueeze(0),tags[i]], ood_params['shot_capacity'], True)  
            if ood_enabled and ood_cache:
                fuzzy_logits = compute_cache_logits(image_features, ood_cache, ood_params['alpha'], ood_params['topk'])
                final_logits -= fuzzy_logits       
            final_logits /= 100     
            final_logits = F.softmax(final_logits/T,dim=-1).cpu().numpy()
            mcm_global_score = np.max(final_logits,axis=1)
            for i in range(len(tags)):
                if tags[i] == 'ID':
                    id_conf = np.concatenate((id_conf, np.array([mcm_global_score[i]])))
                else:
                    ood_conf = np.concatenate((ood_conf, np.array([mcm_global_score[i]])))
        auroc, aupr, fpr = get_measures(id_conf, ood_conf, tpr)
        return auroc, aupr, fpr
