import os
import yaml
import torch
from datasets.imagenet import imagenet_classes
from datasets.imagenet100 import imagenet100_classes
from torchvision import datasets
from torch.utils.data import DataLoader, ConcatDataset,Dataset



# 封装 dataset 加上 source_name
class TaggedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform, source_name):
        super().__init__(root=root, transform=transform)
        self.source_name = source_name

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image, self.source_name
class TaggedCIFAR(Dataset):
    def __init__(self, dataset, source_name):
        self.data = dataset
        self.source_name = source_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        return img, self.source_name

def get_datasets(args, preprocess):
    if args.in_dataset == 'ImageNet' or args.in_dataset == 'ImageNet100':
        if args.in_dataset == 'ImageNet':
            classnames = imagenet_classes
            train_data = datasets.ImageFolder(
                root=os.path.join(args.root_dir, 'ilsvrc2012/train'),
                transform=preprocess
            )
            train_loader = DataLoader(train_data, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
            in_data = TaggedImageFolder(
                root=os.path.join(args.root_dir, 'ilsvrc2012/val'),
                transform=preprocess,
                source_name='ID'
            )
            with open('./configs/imagenet.yaml', 'r') as file:
                cfg = yaml.load(file, Loader=yaml.SafeLoader)
        elif args.in_dataset == 'ImageNet100':  # ImageNet100
            with open('./configs/imagenet100.yaml', 'r') as file:
                cfg = yaml.load(file, Loader=yaml.SafeLoader)
            classnames = imagenet100_classes
            train_data = datasets.ImageFolder(
                root=os.path.join(args.root_dir, 'ImageNet100_MCM/images/train'),
                transform=preprocess
            )
            train_loader = DataLoader(train_data, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
            in_data = TaggedImageFolder(
                root=os.path.join(args.root_dir, 'ImageNet100_MCM/images/val'),
                transform=preprocess,
                source_name='ID'
            )
        iNaturalist_data = TaggedImageFolder(
            root=os.path.join(args.root_dir, 'iNaturalist'),
            transform=preprocess,
            source_name='iNaturalist'
        )
        iNaturalist_dataset = [in_data, iNaturalist_data]
        iNaturalist_dataset = ConcatDataset(iNaturalist_dataset)
        iNaturalist_loader = DataLoader(iNaturalist_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
        SUN_data = TaggedImageFolder(
            root=os.path.join(args.root_dir, 'SUN'),
            transform=preprocess,
            source_name='SUN'
        )
        SUN_dataset = [in_data, SUN_data]
        SUN_dataset = ConcatDataset(SUN_dataset)
        SUN_loader = DataLoader(SUN_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
        Places_data = TaggedImageFolder(
            root=os.path.join(args.root_dir, 'Places'),
            transform=preprocess,
            source_name='Places'
        )
        Places_dataset = [in_data, Places_data]
        Places_dataset = ConcatDataset(Places_dataset)
        Places_loader = DataLoader(Places_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
        Textures_data = TaggedImageFolder(
            root=os.path.join(args.root_dir, 'dtd/images'),
            transform=preprocess,
            source_name='Textures'
        )
        Textures_dataset = [in_data, Textures_data]
        Textures_dataset = ConcatDataset(Textures_dataset)
        Textures_loader = DataLoader(Textures_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
        NINCO_data = TaggedImageFolder(
            root=os.path.join(args.root_dir, 'NINCO/NINCO_OOD_classes'),
            transform=preprocess,
            source_name='NINCO'
        )
        NINCO_dataset = [in_data, NINCO_data]
        NINCO_dataset = ConcatDataset(NINCO_dataset)
        NINCO_loader = DataLoader(NINCO_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
        ssb_hard_data = TaggedImageFolder(
            root=os.path.join(args.root_dir, 'ssb_hard'),
            transform=preprocess,
            source_name='SSB-hard'
        )
        ssb_hard_dataset = [in_data, ssb_hard_data]
        ssb_hard_dataset = ConcatDataset(ssb_hard_dataset)
        ssb_hard_loader = DataLoader(ssb_hard_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
        ood_loaders = {
            'iNaturalist': iNaturalist_loader,
            'SUN': SUN_loader,
            'Places': Places_loader,
            'Textures': Textures_loader,
            'NINCO': NINCO_loader,
            'SSB-hard': ssb_hard_loader
        }
    elif args.in_dataset == 'CIFAR10' or args.in_dataset == 'CIFAR100':
        if args.in_dataset == 'CIFAR10':
            classnames = datasets.CIFAR10.classes
            classnames= [cls.replace('_', ' ') for cls in classnames]
            train_data = datasets.CIFAR10(
                root=args.root_dir,
                train=True,
                transform=preprocess,
                download=True
            )
            test_data = datasets.CIFAR10(root=args.root_dir, train=False, download=True, transform=preprocess)
            train_loader = DataLoader(train_data, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
            in_data = TaggedCIFAR(test_data, "ID")
            with open('./configs/cifar10.yaml', 'r') as file:
                cfg = yaml.load(file, Loader=yaml.SafeLoader)
        elif args.in_dataset == 'CIFAR100':
            classnames = datasets.CIFAR100.classes
            classnames= [cls.replace('_', ' ') for cls in classnames]
            train_data = datasets.CIFAR100(
                root=args.root_dir,
                train=True,
                transform=preprocess,
                download=True
            )
            test_data = datasets.CIFAR100(root=args.root_dir, train=False, download=True, transform=preprocess)
            train_loader = DataLoader(train_data, batch_size=args.test_batch_size, shuffle=False, num_workers=8)
            in_data = TaggedCIFAR(test_data, "ID")
            with open('./configs/cifar100.yaml', 'r') as file:
                cfg = yaml.load(file, Loader=yaml.SafeLoader)
        svhn_dataset = datasets.SVHN(os.path.join(args.root_dir, 'SVHN'), split='test', transform=preprocess, download=True)
        svhn_dataset = [in_data, TaggedCIFAR(svhn_dataset, "OOD")]
        svhn_dataset = ConcatDataset(svhn_dataset)
        svhn_loader = DataLoader(svhn_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
        TinyImageNet_dataset = [in_data,TaggedImageFolder(os.path.join(args.root_dir, 'tin/val/'), preprocess, "OOD")]
        TinyImageNet_dataset = ConcatDataset(TinyImageNet_dataset)
        TinyImageNet_loader = DataLoader(TinyImageNet_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
        LSUN_R_dataset = [in_data,TaggedImageFolder(os.path.join(args.root_dir, 'LSUN_R/'), preprocess, "OOD")]
        LSUN_R_dataset = ConcatDataset(LSUN_R_dataset)
        LSUN_R_loader = DataLoader(LSUN_R_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
        LSUN_C_dataset = [in_data,TaggedImageFolder(os.path.join(args.root_dir, 'LSUN_C/'), preprocess, "OOD")]
        LSUN_C_dataset = ConcatDataset(LSUN_C_dataset)
        LSUN_C_loader = DataLoader(LSUN_C_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
        iSUN_dataset = [in_data, TaggedImageFolder(os.path.join(args.root_dir, 'iSUN/'), preprocess, "OOD")]
        iSUN_dataset = ConcatDataset(iSUN_dataset)
        iSUN_loader = DataLoader(iSUN_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
        Places365_dataset = [in_data,TaggedImageFolder(os.path.join(args.root_dir, 'places365/test_subset/'), preprocess, "OOD")]
        Places365_dataset = ConcatDataset(Places365_dataset)
        Places365_loader = DataLoader(Places365_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
        Textures_dataset = [
            in_data,
            TaggedImageFolder(os.path.join(args.root_dir, 'dtd/images/'), preprocess, "OOD"),
        ]
        Textures_dataset = ConcatDataset(Textures_dataset)
        Textures_loader = DataLoader(Textures_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
        if args.in_dataset == 'CIFAR10':
            CIFAR100_data = datasets.CIFAR100(root=args.root_dir, train=False, download=True, transform=preprocess)
            CIFAR100_dataset = [in_data, TaggedCIFAR(CIFAR100_data, "OOD")]
            CIFAR100_dataset = ConcatDataset(CIFAR100_dataset)
            CIFAR100_loader = DataLoader(CIFAR100_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
            ood_loaders = {
                'SVHN': svhn_loader,
                'TinyImageNet': TinyImageNet_loader,
                'LSUN_R': LSUN_R_loader,
                'LSUN_C': LSUN_C_loader,
                'iSUN': iSUN_loader,
                'Places365': Places365_loader,
                'Textures': Textures_loader,
                'CIFAR100': CIFAR100_loader
            }
        else:  # CIFAR100
            CIFAR10_data = datasets.CIFAR10(root=args.root_dir, train=False, download=True, transform=preprocess)
            CIFAR10_dataset = [in_data, TaggedCIFAR(CIFAR10_data, "OOD")]
            CIFAR10_dataset = ConcatDataset(CIFAR10_dataset)
            CIFAR10_loader = DataLoader(CIFAR10_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=8)
            ood_loaders = {
                'SVHN': svhn_loader,
                'TinyImageNet': TinyImageNet_loader,
                'LSUN_R': LSUN_R_loader,
                'LSUN_C': LSUN_C_loader,
                'iSUN': iSUN_loader,
                'Places365': Places365_loader,
                'Textures': Textures_loader,
                'CIFAR10': CIFAR10_loader
            } 
        
    return train_loader, ood_loaders, classnames, cfg
        