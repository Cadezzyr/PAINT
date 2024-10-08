import logging
import random
import torch
import torch.optim as optim
import torch.nn as nn
from robustbench.data import load_imagenetc
from robustbench.data import load_cifar100
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
# from robustbench.utils import clean_accuracy_vit as accuracy
import torchvision.transforms
import my_transforms as my_transforms
import numpy as np
import os
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import torch
from robustbench.model_zoo.architectures import vision_transformer
from robustbench.model_zoo.architectures import prompt
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import torchvision.transforms as transforms




def mixup(batch_x, batch_y, alpha=0.5, use_cuda=False):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = batch_x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
        # print(lam * batch_y + (1 - lam) * batch_y[index, :])
    else:
        index = torch.randperm(batch_size)
    return lam * batch_x + (1 - lam) * batch_x[index, :] ,batch_y,batch_y[index],lam


def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (224, 224, 3)
    n_pixels = img_shape[0]
    clip_min, clip_max = 0.0, 1.0
    p_hflip = 0.5
    tta_transforms = transforms.Compose([
        my_transforms.Clip(0.0, 1.0),
        my_transforms.ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        # transforms.Pad(padding=int(n_pixels / 2), padding_mode='edge'),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            resample=PIL.Image.BILINEAR,
            fillcolor=None
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
        transforms.CenterCrop(size=n_pixels),
        transforms.RandomHorizontalFlip(p=p_hflip),
        my_transforms.GaussianNoise(0, gaussian_std),
        my_transforms.Clip(clip_min, clip_max)
    ])
    return tta_transforms

def get_common_transform():
    transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.ToTensor()
    ])
    return transform

def return_high_confidence_sample(P_L):
    sample_list =[]
    low_sample_list = []
    P_L_S = torch.nn.Softmax(dim=1)(P_L)
    for instance in range(50):
        if float(P_L_S[instance].max(0).values)>= 0.6  and float(sorted(P_L_S[instance])[-2])<=0.3:
            sample_list.append(int(instance))
        else:
            low_sample_list.append(int(instance))
    return torch.tensor(sample_list).type(torch.int64),torch.tensor(low_sample_list).type(torch.int64)

# Fix random seed for reproducibility
def same_seeds(seed):
	  torch.manual_seed(seed)
	  if torch.cuda.is_available():
		    torch.cuda.manual_seed(seed)
		    torch.cuda.manual_seed_all(seed)
	  np.random.seed(seed)
	  random.seed(seed)
	  torch.backends.cudnn.benchmark = False
	  torch.backends.cudnn.deterministic = True

def main():
    transforms_cotta = get_tta_transforms()
    original_model = create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=1000,
        drop_rate=0,
        drop_path_rate=0,
        drop_block_rate=None,
        img_size=224,
        prompt_pool =False,
        pool_size =None,
        prompt_length = None,

    )


    model = create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=1000,
        drop_rate=0,
        drop_path_rate=0,
        drop_block_rate=None,
        prompt_length=2,
        embedding_key='cls',
        prompt_init='zero',
        prompt_pool=True,
        prompt_key=True,
        pool_size=5,
        top_k=1,
        batchwise_prompt=True,
        prompt_key_init='uniform',
        head_type='token+prompt',
        use_prompt_mask=False,
        deep_prompt = False,
        img_size=224,

    )
    transform=get_common_transform()
    same_seeds(0)
    for p in original_model.parameters():
        p.requires_grad = False

    # freeze args.freeze[blocks, patch_embed, cls_token] parameters
    for n, p in model.named_parameters():
        if n.startswith(tuple(['patch_embed','blocks.3','blocks.4','blocks.5','blocks.6','blocks.7','blocks.8','blocks.9','blocks.10','blocks.11','norm', 'pos_embed','head','cls_token']))or n.endswith('prompt_key'):
            # print(n)
            p.requires_grad = False
        else:
            print(n)
    model = model.cuda(1)
    original_model = original_model.cuda(1)
    corruptions = ['gaussian_noise','shot_noise','impulse_noise','defocus_blur','glass_blur','motion_blur','zoom_blur','snow',
                    'frost','fog','brightness','contrast','elastic_transform','pixelate','jpeg_compression']
    corruptions_1 = ['gaussian_noise','shot_noise','impulse_noise' ]
    corruptions_2 = ['defocus_blur','glass_blur','motion_blur','zoom_blur']
    corruptions_3 = ['snow','frost','fog','brightness']
    corruptions_4 = ['contrast','elastic_transform','pixelate','jpeg_compression']
    # corruptions = ['fog']
    severitys = [5]
    for corruption in corruptions:
        for severity in severitys:
            x_test, y_test = load_imagenetc(n_examples=5000, corruptions=[corruption], severity=severity ,shuffle=False)
            acc =0
            for i in range(100):
                x_test_batch =x_test[i*50:(i+1)*50,]
                y_test_batch = y_test[i*50:(i+1)*50,]
                x_test_batch,y_test_batch = x_test_batch.cuda(1),y_test_batch.cuda(1)
                x_test_aug = transforms_cotta(x_test_batch)
                x_test_aug = x_test_aug.cuda(1)
                with torch.no_grad():
                    cls_features = original_model(transform(x_test_batch))
                    cls_features = cls_features['pre_logits']
                y_batch =model(transform(x_test_batch),cls_features=cls_features)
                x_test_aug = x_test_aug.float()
                # onehot operation.
                one_hot = torch.tensor(np.eye(1000)[np.array((y_batch['logits'].max(1)[1]).cpu())], device='cuda:1')
                # Screening for high confidence samples
                sample_list_aug,low_sample_list_aug = return_high_confidence_sample(y_batch['logits'])
                softmax_out = nn.Softmax(dim=1)(y_batch['logits'])
                # Fine-tuning of the first few layers of the model as well
                optimizer = optim.Adam([
                    {'params': [
                                model.blocks[0].attn.qkv.weight,
                                model.blocks[0].attn.qkv.bias, model.blocks[0].attn.proj.weight,
                                model.blocks[0].attn.proj.bias, model.blocks[0].norm2.weight,
                                model.blocks[0].norm2.bias, model.blocks[0].mlp.fc1.weight,
                                model.blocks[0].mlp.fc1.bias, model.blocks[0].mlp.fc2.weight,
                                model.blocks[0].mlp.fc2.bias,
                                model.blocks[1].norm1.weight, model.blocks[1].norm1.bias,
                                model.blocks[1].attn.qkv.weight, model.blocks[1].attn.qkv.bias,
                                model.blocks[1].attn.proj.weight, model.blocks[1].attn.proj.bias,
                                model.blocks[1].norm2.weight, model.blocks[1].norm2.bias,
                                model.blocks[1].mlp.fc1.weight, model.blocks[1].mlp.fc1.bias,
                                model.blocks[1].mlp.fc2.weight, model.blocks[1].mlp.fc2.bias,
                                model.blocks[2].norm1.weight, model.blocks[2].norm1.bias,
                                model.blocks[2].attn.qkv.weight, model.blocks[2].attn.qkv.bias,
                                model.blocks[2].attn.proj.weight, model.blocks[2].attn.proj.bias,
                                model.blocks[2].norm2.weight, model.blocks[2].norm2.bias,
                                model.blocks[2].mlp.fc1.weight, model.blocks[2].mlp.fc1.bias,
                                model.blocks[2].mlp.fc2.weight, model.blocks[2].mlp.fc2.bias,
                                ], 'lr':
                        2.5e-6},
                    {'params': [model.prompt.prompt], 'lr': 0.075}])
                #Mixup operations on pseudo-labels and images
                mixup_aug,mixup_one_hot_a,mixup_one_hot_b,lam= mixup(x_test_aug[sample_list_aug],one_hot[sample_list_aug])
                y_batch_mixup = model(mixup_aug,cls_features =cls_features)
                softmax_out_mix = nn.Softmax(dim=1)(y_batch_mixup['logits'])
                loss_ent = nn.CrossEntropyLoss()(softmax_out,softmax_out)
                loss_div = nn.CrossEntropyLoss()((torch.mean(softmax_out,dim=0)).reshape(1,1000),torch.mean(softmax_out,dim=0).reshape(1,1000))
                loss_ic = lam*nn.CrossEntropyLoss()(softmax_out_mix,mixup_one_hot_a)+(1-lam)*nn.CrossEntropyLoss()(softmax_out_mix,mixup_one_hot_b)
                loss =loss_ent-loss_div+loss_ic
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    y_batch = model((x_test_batch),cls_features=cls_features)
                acc+= (y_batch['logits'].max(1)[1] == y_test_batch).float().sum()
            acc=acc.item() / x_test.shape[0]
            print('__________________________________________________________'+corruption+str(severity)+'____________________________________________________')
            print((acc))

    for n, p in model.named_parameters():
        if str(n) == 'prompt.prompt':
            print(n, p)
            print(p.shape)


if __name__ == '__main__':
    main()

