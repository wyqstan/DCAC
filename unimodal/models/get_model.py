import torch
import torch.nn as nn


def get_model(args):
    model = None

    if args.model == "resnet":
        from models.imagenet.resnet import resnet50
        if args.in_dataset in ['cifar10', 'cifar100']:
            
            model = resnet50(num_classes=args.num_classes)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
            if args.in_dataset == 'cifar10':
                model.load_state_dict(torch.load('./checkpoints/best_resnet50_cifar10.pth'))
            elif args.in_dataset == 'cifar100':
                model.load_state_dict(torch.load('./checkpoints/best_resnet50_cifar100.pth'))
        else:
            model = resnet50(num_classes=args.num_classes, pretrained=True)

    elif args.model == "densenet":
        import models.imagenet.densenet as densenet
        model = densenet.densenet201(weights='IMAGENET1K_V1')

    elif args.model == "vit":
        import models.imagenet.vision_transformer as vit
        model = vit.vit_b_16(weights='IMAGENET1K_V1')

    elif args.model == "swin":
        import models.imagenet.swin_transformer as swin
        model = swin.swin_b(weights='IMAGENET1K_V1')

    elif args.model == "convnext":
        import models.imagenet.convnext as convnext
        model = convnext.convnext_base(weights='IMAGENET1K_V1')

    elif args.model == "regnet":
        import models.imagenet.regnet as regnet
        model = regnet.regnet_x_8gf(weights='IMAGENET1K_V1')

    elif args.model == "efficientnet":
        import models.imagenet.efficientnet as efficientnet
        model = efficientnet.efficientnet_v2_m(weights='IMAGENET1K_V1')
    
    elif args.model == "maxvit":
        import models.imagenet.maxvit as maxvit
        model = maxvit.maxvit_t(weights='IMAGENET1K_V1')

    return model


