import sys
sys.path.append('..')
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose, Normalize
from einops import repeat
import clip
from tqdm import tqdm
import math
import numpy as np
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
def get_entropy(loss, num_class):
    max_entropy = math.log2(num_class)
    return loss / max_entropy
class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model,num_neg_prompts=300,N_CTX=16,CTX_INIT = "",CSC=True,CLASS_TOKEN_POSITION="end"):
        super().__init__()
        n_cls = len(classnames)
        self.num_neg_prompts = num_neg_prompts
        self.num_local_prompts = n_cls
        n_ctx = N_CTX
        ctx_init = CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        
        # for global prompt initialization: frozen and hand-crafted 'a photo of {c}'
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            global_ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                global_ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                global_ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(global_ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial global prompt context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        global_prompts = ["a photo of a" + " " + name + "." for name in classnames]
        
        global_tokenized_prompts = torch.cat([clip.tokenize(p).cuda() for p in global_prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(global_tokenized_prompts).type(dtype)
        
        self.global_embedding = embedding
        self.global_tokenized_prompts = global_tokenized_prompts  # torch.Tensor  #1000,77
        self.class_token_position = CLASS_TOKEN_POSITION

        # for local prompt initialization: learnable
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            local_ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                local_ctx_vectors = torch.empty(self.num_local_prompts, n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(local_ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial local context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.local_ctx = nn.Parameter(local_ctx_vectors)  # to be optimized
        
        local_prompts = [prompt_prefix + " " + name + "." for name in classnames]
        local_tokenized_prompts = torch.cat([clip.tokenize(p).cuda() for p in local_prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(local_tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.local_tokenized_prompts = local_tokenized_prompts

        # for local prompt initialization: learnable and random initialization
        print("Initializing negative local contexts")
        neg_ctx_vectors = torch.empty(self.num_neg_prompts, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(neg_ctx_vectors, std=0.02)
        neg_prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial local context: "{neg_prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.neg_ctx = nn.Parameter(neg_ctx_vectors)  # to be optimized
         
        neg_prompts = [neg_prompt_prefix + " " + "." for _ in range(self.num_neg_prompts)]
        neg_tokenized_prompts = torch.cat([clip.tokenize(p).cuda() for p in neg_prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(neg_tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        
        self.register_buffer("neg_token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("neg_token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS
        
        self.neg_tokenized_prompts = neg_tokenized_prompts

    def forward(self):
        assert self.class_token_position == 'end', 'not expected class token position.'
        
        local_ctx = self.local_ctx #100,16,512
        if local_ctx.dim() == 2:
            local_ctx = local_ctx.unsqueeze(0).expand(self.num_ood_prompts, -1, -1)

        prefix = self.token_prefix #100,1,512
        suffix = self.token_suffix #1000,60,512

        local_prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                local_ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )
        
        
        if local_ctx.dim() == 2:
            local_ctx = local_ctx.unsqueeze(0).expand(self.num_neg_prompts, -1, -1)

        neg_prefix = self.neg_token_prefix #100,1,512
        neg_suffix = self.neg_token_suffix #1000,60,512

        neg_prompts = torch.cat(
            [
                neg_prefix,  # (n_cls, 1, dim)
                self.neg_ctx,     # (n_cls, n_ctx, dim)
                neg_suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        )


        return self.global_embedding, local_prompts, neg_prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _, _, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    

class CustomCLIP(nn.Module):
    def __init__(self,  classnames, clip_model):
        super().__init__()
        self.classnames = classnames
        self.prompt_learner = PromptLearner(classnames, clip_model)
        self.global_tokenized_prompts = self.prompt_learner.global_tokenized_prompts
        self.local_tokenized_prompts =self.prompt_learner.local_tokenized_prompts
        self.neg_tokenized_prompts = self.prompt_learner.neg_tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
    
    def multi_loader_select(self, images, label):
        '''
        Global Prompt Guided Negative Augmentation
        select images according to similarity between global image features label texts.
        '''
        with torch.no_grad():
            image_features, local_image_features = [], []
            similarity_list = []
            
            for image in images:
                image_feature, local_image_feature = self.image_encoder(image.type(self.dtype))
                image_features.append(image_feature)
                local_image_features.append(local_image_feature)

            global_prompts, _, _ = self.prompt_learner()
            global_tokenized_prompts = self.global_tokenized_prompts
            global_text_features = self.text_encoder(global_prompts, global_tokenized_prompts)
            global_text_features = global_text_features / global_text_features.norm(dim=-1, keepdim=True)

            image_features = [image_feature / image_feature.norm(dim=-1, keepdim=True) for image_feature in image_features]
            global_text_selected_label = global_text_features.gather(0, label[...,None].expand_as(image_features[0]))

            for image_feature in image_features:
                similarity_list.append(torch.nn.functional.cosine_similarity(image_feature, global_text_selected_label, dim=-1))

            image_features = torch.stack(image_features,dim=0)
            local_image_features = torch.stack(local_image_features,dim=0)
            similarity_list = torch.stack(similarity_list,dim=0)
            
            return similarity_list, image_features, local_image_features

    def forward(self, images, image_features=None, local_image_features=None, max_list=None, min_list=None):
        if self.training:
            num_region, dimension = local_image_features.shape[-2:]

            _, local_prompts, neg_prompts = self.prompt_learner()

            local_tokenized_prompts = self.local_tokenized_prompts
            neg_tokenized_prompts = self.neg_tokenized_prompts

            local_text_features = self.text_encoder(local_prompts, local_tokenized_prompts)
            neg_text_features = self.text_encoder(neg_prompts, neg_tokenized_prompts)
            
            # positive and negative feature selection
            with torch.no_grad():
                pos_local_image_features = local_image_features.gather(0,repeat(max_list, 'q b -> q b n c', n=num_region, c = dimension)).squeeze()
                neg_local_image_features = local_image_features.gather(0,repeat(min_list, 'q b -> q b n c', n=num_region, c = dimension)).squeeze()

            pos_local_image_features = pos_local_image_features / pos_local_image_features.norm(dim=-1, keepdim=True)
            neg_local_image_features = neg_local_image_features / neg_local_image_features.norm(dim=-1, keepdim=True)

            local_text_features = local_text_features / local_text_features.norm(dim=-1, keepdim=True)
            neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()
            logits_local = logit_scale * pos_local_image_features @ local_text_features.t()
            p2n_logits_local = logit_scale * neg_local_image_features @ local_text_features.t()
            n2p_logits_local = logit_scale * pos_local_image_features @ neg_text_features.t()
            neg_logits_local = logit_scale * neg_local_image_features @ neg_text_features.t()

            # for diversity regularization
            loss_div = torch.nn.functional.cosine_similarity(local_text_features[None,:,:], local_text_features[:,None,:], dim=-1)

            loss_div = torch.sum(loss_div,dim=-1)/self.prompt_learner.num_neg_prompts
            loss_div = torch.sum(loss_div,dim=-1)/(self.prompt_learner.num_neg_prompts-1)

            return logits_local, p2n_logits_local, n2p_logits_local, neg_logits_local, loss_div

        else: # for inference
            global_prompts, local_prompts, neg_prompts = self.prompt_learner()

            global_tokenized_prompts = self.global_tokenized_prompts
            local_tokenized_prompts = self.local_tokenized_prompts
            neg_tokenized_prompts = self.neg_tokenized_prompts

            global_text_features = self.text_encoder(global_prompts, global_tokenized_prompts)
            local_text_features = self.text_encoder(local_prompts, local_tokenized_prompts)
            neg_text_features = self.text_encoder(neg_prompts, neg_tokenized_prompts)
            
            image_features, local_image_features = self.image_encoder(images.type(self.dtype))
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
            
            global_text_features = global_text_features / global_text_features.norm(dim=-1, keepdim=True)
            local_text_features = local_text_features / local_text_features.norm(dim=-1, keepdim=True)
            neg_text_features = neg_text_features / neg_text_features.norm(dim=-1, keepdim=True)

            logit_scale = self.logit_scale.exp()

            logits = logit_scale * image_features @ global_text_features.t()
            logits_local = logit_scale * local_image_features @ local_text_features.t()
            neg_logits_local = logit_scale * local_image_features @ neg_text_features.t()
            
            return logits, logits_local, neg_logits_local, image_features
    
    def get_train_id_entropy(self, train_loader, device):
        id_entropy = np.array([])
        with torch.no_grad():
            for i, (images,targets) in enumerate(tqdm(train_loader)):
                images = images.to(device)
                output, output_local, neg_output_local, image_features = self.forward(images)
                loss = softmax_entropy(output)
                prop_entropy = get_entropy(loss, len(self.classnames))
                id_entropy = np.concatenate((id_entropy, prop_entropy.cpu().numpy()))
        return id_entropy