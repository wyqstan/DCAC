import torch
import math
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from methods.utils import update_cache, compute_cache_logits,get_measures
def get_entropy(loss, num_class):
    max_entropy = math.log2(num_class)
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


def run_test_local_prompt(ood_cfg, loader,model, id_entropy):
    with torch.no_grad():
        ood_cache = {}
        #Unpack all hyperparameters
        ood_enabled = ood_cfg['enabled']
        if ood_cfg:
            ood_params = {k: ood_cfg[k] for k in ['shot_capacity', 'alpha', 'lower',  'topk']}
        lower_threshold = np.percentile(id_entropy, ood_params['lower'])
        id_conf_gl = np.array([])
        ood_conf_gl = np.array([])
        # data_record = []
        T = 1
        tpr = 0.95
        for images, tags in tqdm(loader):
            images = images.cuda()
            output, output_local, neg_output_local, image_features = model(images)
            loss = softmax_entropy(output)
            prob_map = output.softmax(1)
            pred = torch.argmax(output,dim=1)
            num_classes = prob_map.shape[1]
            prop_entropy = get_entropy(loss, num_classes)
            final_logits = output.clone()
            for i in range(prob_map.shape[0]):
                if ood_enabled and lower_threshold < prop_entropy[i].item():
                    update_cache(ood_cache, int(pred[i].item()), [image_features[i].unsqueeze(0), loss[i].item(), prob_map[i].unsqueeze(0),tags[i]], ood_params['shot_capacity'], True)  
            if ood_enabled and ood_cache:
                fuzzy_logits = compute_cache_logits(image_features, ood_cache, ood_params['alpha'], ood_params['topk'])
                final_logits -= fuzzy_logits
            final_logits /= 100.0
            output_local /= 100.0
            neg_output_local /= 100.0
            smax_global = F.softmax(final_logits/T, dim=-1).cpu().numpy()
            mcm_global_score = np.max(smax_global, axis=1)
            N, C = output_local.shape[1:]
            smax_local = torch.topk((torch.exp(output_local/T)/ \
                torch.sum(torch.exp(torch.cat((output_local, neg_output_local),dim=-1)/T),dim=-1,keepdim=True)).reshape(-1, N*C), k=10, dim=-1)[0]
            mcm_local_score= torch.mean(smax_local,dim=1).cpu().numpy()
            for i in range(len(tags)):
                if tags[i] == 'ID':
                    id_conf_gl = np.concatenate((id_conf_gl, np.array([mcm_global_score[i] + mcm_local_score[i]])))
                else:
                    ood_conf_gl = np.concatenate((ood_conf_gl, np.array([mcm_global_score[i] + mcm_local_score[i]])))

        auroc_gl, aupr_gl, fpr_gl = get_measures(id_conf_gl, ood_conf_gl, tpr)
        return  auroc_gl, aupr_gl, fpr_gl