import sys
import clip
import torch
import os
import yaml
from models.coop import CustomCLIP as CustomCLIP_Coop
from models.local_prompt import CustomCLIP as CustomCLIP_LocalPrompt

def clip_classifier_coop(model,clip_model):
    with torch.no_grad():
        prompts = model.prompt_learner()
        tokenized_prompts = model.tokenized_prompts
        text_features = model.text_encoder(prompts, tokenized_prompts).type(clip_model.dtype)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features
def get_model(args,device):
    model = None
    clip_model, preprocess = clip.load(args.model, device=device)
    clip_model.eval()
    if args.method == 'mcm':
        model = CustomCLIP_Coop(args.classnames,clip_model,"a photo of a",False, 4)
    if args.method == 'coop':
        model = CustomCLIP_Coop(args.classnames,clip_model,False,False,16).to(device)
        pretrained_ctx = torch.load(args.model_dir)['state_dict']
        with torch.no_grad():
            model.prompt_learner.load_state_dict(pretrained_ctx)
    elif args.method == 'locoop':
        model = CustomCLIP_Coop(args.classnames,clip_model,False,False,16).to(device)
        pretrained_ctx = torch.load(args.model_dir)['state_dict']
        with torch.no_grad():
            model.prompt_learner.load_state_dict(pretrained_ctx)
    elif args.method == 'sct':
        model = CustomCLIP_Coop(args.classnames,clip_model,False,False,16).to(device)
        pretrained_ctx = torch.load(args.model_dir)['state_dict']
        with torch.no_grad():
            model.prompt_learner.load_state_dict(pretrained_ctx)
    elif args.method == 'ospcoop':
        model = CustomCLIP_Coop(args.classnames,clip_model,False,False,16).to(device)
        pretrained_ctx = torch.load(args.model_dir)['state_dict']
        with torch.no_grad():
            model.prompt_learner.load_state_dict(pretrained_ctx)
    elif args.method == 'local-prompt':
        model = CustomCLIP_LocalPrompt(args.classnames,clip_model).to(device)
        pretrained_ctx = torch.load(args.model_dir)['state_dict']
        with torch.no_grad():
            model.prompt_learner.load_state_dict(pretrained_ctx)
    if args.method in ['mcm','coop','locoop', 'sct', 'ospcoop']:
        clip_weights = clip_classifier_coop(model,clip_model)
        return model,clip_weights
    else:
        return model, None