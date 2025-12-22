import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose, Normalize
import math
import clip
from tqdm import tqdm
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
def get_entropy(loss, num_class):
    max_entropy = math.log2(num_class)
    return loss / max_entropy
class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, N_CTX = 16, CTX_INIT = "a photo of a", CSC = False, CLASS_TOKEN_POSITION = "end"):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = N_CTX
        ctx_init = CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" ")) # 4
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        print("------------------")
        # print(self.ctx.shape)
        print("------------------")        

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(clip_model.positional_embedding.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts.type(self.dtype) + self.positional_embedding.to(dtype = self.dtype, device = prompts.device)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, _, _, _ = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class CustomCLIP(nn.Module):
    def __init__(self, classnames, 
                 clip_model,
                 CTX_INIT = "A photo of ", 
                 single = False, 
                 n_ctx = 16):
        super().__init__()
        self.classnames = classnames
        self.prompt_learner = PromptLearner(classnames, clip_model, CTX_INIT=CTX_INIT, N_CTX=n_ctx)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.frozen_weights()
        self.CLIP_Transforms = Compose([
                            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                        ])

        self.single = single
        
    def frozen_weights(self):
        print("trainable parameters:")
        for name, param in self.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
            else:
                print(name)
                param.requires_grad_(True)

        print("---------------------------")
        
    def forward(self, image):
        if self.single : image = self.CLIP_Transforms(image)

        image_features, local_image_features = self.image_encoder(image.type(self.dtype))
        prompts = self.prompt_learner().to(image.device)
        tokenized_prompts = self.tokenized_prompts.to(image.device)
        text_features = self.text_encoder(prompts, tokenized_prompts).type(self.dtype)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        local_image_features = local_image_features / local_image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        logits = logit_scale * image_features @ text_features.transpose(-1, -2)
        logits_local =  logit_scale * local_image_features @ text_features.T
        
        return logits, logits_local
    def get_train_id_entropy(self, train_loader, device):
        id_entropy = np.array([])
        with torch.no_grad():
            for i, (images,targets) in enumerate(tqdm(train_loader)):
                images = images.to(device)
                image_features, local_image_features = self.image_encoder(images.type(self.dtype))
                prompts = self.prompt_learner().to(images.device)
                tokenized_prompts = self.tokenized_prompts.to(images.device)
                text_features = self.text_encoder(prompts, tokenized_prompts).type(self.dtype)
                logit_scale = self.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.T
                loss = softmax_entropy(logits)
                entropy = get_entropy(loss, len(self.classnames))
                id_entropy = np.concatenate((id_entropy, entropy.cpu().numpy()))
        return id_entropy

