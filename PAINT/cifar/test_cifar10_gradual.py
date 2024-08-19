import logging
import random

import torch
import torch.optim as optim
import torch.nn as nn
from robustbench.data import load_cifar10c
from robustbench.data import load_cifar10
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy_vit as accuracy
import my_transforms as my_transforms
import numpy as np
import os
import PIL
from PIL import Image
import matplotlib.pyplot as plt
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
# from conf import cfg, load_cfg_fom_args
from robustbench.model_zoo.architectures import vision_transformer
from robustbench.model_zoo.architectures import prompt
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import torchvision.transforms as transforms
parameters_pool =dict()
result_pool =dict()

def criterion(batch_x, batch_y, alpha=0.1, use_cuda=False):

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        # print(lam)
    else:
        lam = 1
    batch_size = batch_x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
        # print(lam * batch_y + (1 - lam) * batch_y[index, :])
    else:
        index = torch.randperm(batch_size)
        # mixed_batchx = lam * batch_x + (1 - lam) * batch_x[index, :]
        # mixed_batchy = lam * batch_y +(1-lam) * batch_y[index,:]
    return lam * batch_x + (1 - lam) * batch_x[index, :] ,batch_y,batch_y[index],lam
def get_tta_transforms(gaussian_std: float=0.005, soft=False, clip_inputs=False):
    img_shape = (32, 32, 3)
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
        if float(P_L_S[instance].max(0).values)>= 0.7 and float(sorted(P_L_S[instance])[-2])<=0.1:
            sample_list.append(int(instance))
        else:
            low_sample_list.append(int(instance))
    return torch.tensor(sample_list).type(torch.int64),torch.tensor(low_sample_list).type(torch.int64)
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transforms_cotta = get_tta_transforms()
    original_model = create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=10,
        drop_rate=0,
        drop_path_rate=0,
        drop_block_rate=None,
        img_size=224,
        prompt_pool =False,
        pool_size =None,
        prompt_length = None,
        checkpoint_path ='/home/yrzhen/Vision-Transformer-ViT-master/output/0.9835000038146973.pt'
    )
    model = create_model(
        'vit_base_patch16_224',
        pretrained=True,
        num_classes=10,
        drop_rate=0,
        drop_path_rate=0,
        drop_block_rate=None,
        prompt_length=3,
        embedding_key='cls',
        prompt_init='zero',
        prompt_pool=True,
        prompt_key=True,
        pool_size=5 ,
        top_k=1,
        batchwise_prompt=True,
        prompt_key_init='uniform',
        head_type='token+prompt',
        use_prompt_mask=False,
        img_size=224,
        checkpoint_path='0.9835000038146973_3prompt.pth.tar'
    )
    # fill_tensor  = torch.randn((1,6,768),device=device)
    # pretrained_dict = torch.load('/home/yrzhen/Vision-Transformer-ViT-master/output/0.9835000038146973.pt')
    # for k,v in pretrained_dict.items():
    #     # print(k)
    #     if k == 'pos_embed':
    #         # print(v['pos_embed'])
    #         v =v.cuda()
    #         pretrained_dict[k] =torch.concat([v,fill_tensor],dim=1)
    #         print('saved')
    #         print(v.shape)
    # # for k,v in pretrained_dict.items():
    # #     # print(k)
    # #     if k == 'pos_embed':
    # #         # print(v['pos_embed'])
    # #         v =v.cuda()
    # #         v=torch.concat([v,fill_tensor],dim=1)
    # #         print('saved')
    # #         print(v.shape)
    # torch.save(pretrained_dict, '0.9835000038146973_6prompt.pth.tar')


    transform=get_common_transform()
    same_seeds(0)
    for p in original_model.parameters():
        p.requires_grad = False

    # freeze other parameters
    for n, p in model.named_parameters():
        if n.startswith(tuple(['patch_embed','blocks.3','blocks.4','blocks.5','blocks.6','blocks.7','blocks.8','blocks.9','blocks.10','blocks.11', 'norm', 'pos_embed','head',]))or n.endswith('prompt_key'):      ##head  blocks
            p.requires_grad = False
        else:
            print(n)

    model = model.cuda(1)
    original_model = original_model.cuda(1)

    corruptions_0 = ['gaussian_noise','shot_noise','impulse_noise','defocus_blur','glass_blur','motion_blur','zoom_blur','snow',
                    'frost','fog','brightness','contrast','elastic_transform','pixelate','jpeg_compression']
    corruptions_1 = ['jpeg_compression','shot_noise','zoom_blur','frost','contrast','fog','defocus_blur','elastic_transform','gaussian_noise',
                     'brightness','glass_blur','impulse_noise','pixelate','snow','motion_blur']
    corruptions_2 = ['contrast','defocus_blur','gaussian_noise','shot_noise','snow','frost','glass_blur','zoom_blur','elastic_transform',
                     'jpeg_compression','pixelate','brightness','impulse_noise','motion_blur','fog']
    corruptions_3 = ['shot_noise','fog','glass_blur','pixelate','snow','elastic_transform','brightness','impulse_noise','defocus_blur',
                     'frost','contrast','gaussian_noise','motion_blur','jpeg_compression','zoom_blur']
    corruptions_4 = ['pixelate','glass_blur','zoom_blur','snow','fog','impulse_noise','brightness','motion_blur','frost','jpeg_compression',
                     'gaussian_noise','shot_noise','contrast','defocus_blur','elastic_transform']
    corruptions_5 = ['motion_blur','snow','fog','shot_noise','defocus_blur','contrast','zoom_blur','brightness','frost','elastic_transform',
                     'glass_blur','gaussian_noise','pixelate','jpeg_compression','impulse_noise']
    corruptions_6 = ['frost','impulse_noise','jpeg_compression','contrast','zoom_blur','glass_blur','pixelate','snow','defocus_blur','motion_blur','brightness','elastic_transform','shot_noise','fog','gaussian_noise']
    corruptions_7 = ['glass_blur','zoom_blur','impulse_noise','fog','snow','jpeg_compression','gaussian_noise','frost','shot_noise','brightness','contrast','motion_blur','pixelate','defocus_blur','elastic_transform']
    corruptions_8 = ['defocus_blur','motion_blur','zoom_blur','shot_noise','gaussian_noise','glass_blur','jpeg_compression','fog','contrast','pixelate','frost','snow','brightness','elastic_transform','impulse_noise']
    corruptions_9 = ['contrast','gaussian_noise','defocus_blur','zoom_blur','frost','glass_blur','jpeg_compression','fog','pixelate','elastic_transform','shot_noise','impulse_noise','snow','motion_blur','brightness']
    # corruptions = ['pixelate','jpeg_compression']
    severitys = [1,2,3,4,5,4,3,2,1]
    for corruption in corruptions_0:
        for severity in severitys:
            x_test, y_test = load_cifar10c(n_examples=10000, corruptions=[corruption], severity=severity ,shuffle=False)
            print(x_test.shape)
            acc =0
            for i in range(200):
                x_test_batch =x_test[i*50:(i+1)*50,]
                y_test_batch = y_test[i*50:(i+1)*50,]
                # print(x_test_batch.shape,y_test_batch.shape)

                x_test_batch,y_test_batch = x_test_batch.cuda(1),y_test_batch.cuda(1)
                # x_test_aug = aug_dispoal(x_test_batch)
                x_test_aug = transforms_cotta(x_test_batch)
                x_test_aug = x_test_aug.cuda(1)

                cls_features = original_model(transform(x_test_batch))
                cls_features = cls_features['pre_logits']
                y_batch =model(transform(x_test_batch),cls_features=cls_features)
                x_test_aug = x_test_aug.float()
                # onehot operation.
                one_hot = torch.tensor(np.eye(10)[np.array((y_batch['logits'].max(1)[1]).cpu())], device='cuda:1')
                # Screening for high confidence samples
                sample_list_aug,low_sample_list_aug = return_high_confidence_sample(y_batch['logits'])
                # Mixup operations on pseudo-labels and images
                mixup_aug,mixup_one_hot_a,mixup_one_hot_b,lam= criterion(x_test_aug[sample_list_aug],one_hot[sample_list_aug])
                y_batch_mixup = model(transform(mixup_aug),cls_features =cls_features)
                softmax_out_mix = nn.Softmax(dim=1)(y_batch_mixup['logits'])
                softmax_out = nn.Softmax(dim=1)(y_batch['logits'])
                # Fine-tuning of the first few layers of the model as well
                optimizer = optim.Adam([
                    {'params': [model.blocks[0].norm1.bias,model.blocks[0].attn.qkv.weight,model.blocks[0].attn.qkv.bias,model.blocks[0].attn.proj.weight,model.blocks[0].attn.proj.bias,model.blocks[0].norm2.weight,model.blocks[0].norm2.bias,model.blocks[0].mlp.fc1.weight,model.blocks[0].mlp.fc1.bias,model.blocks[0].mlp.fc2.weight,model.blocks[0].mlp.fc2.bias,
                                model.blocks[1].norm1.weight,  model.blocks[1].norm1.bias,model.blocks[1].attn.qkv.weight,model.blocks[1].attn.qkv.bias,model.blocks[1].attn.proj.weight,model.blocks[1].attn.proj.bias,model.blocks[1].norm2.weight,model.blocks[1].norm2.bias,model.blocks[1].mlp.fc1.weight,model.blocks[1].mlp.fc1.bias,model.blocks[1].mlp.fc2.weight,model.blocks[1].mlp.fc2.bias,
                                model.blocks[2].norm1.weight,  model.blocks[2].norm1.bias,model.blocks[2].attn.qkv.weight,model.blocks[2].attn.qkv.bias,model.blocks[2].attn.proj.weight,model.blocks[2].attn.proj.bias,model.blocks[2].norm2.weight,model.blocks[2].norm2.bias,model.blocks[2].mlp.fc1.weight,model.blocks[2].mlp.fc1.bias,model.blocks[2].mlp.fc2.weight,model.blocks[2].mlp.fc2.bias,
                                ],'lr': 1e-5},
                    {'params': [model.cls_token,model.prompt.prompt], 'lr': 0.05}])
                loss_3 = nn.CrossEntropyLoss()(softmax_out,softmax_out)
                loss_2 = nn.CrossEntropyLoss()((torch.mean(softmax_out,dim=0)).reshape(1,10),torch.mean(softmax_out,dim=0).reshape(1,10))
                loss_7 = lam * nn.CrossEntropyLoss()(softmax_out_mix, mixup_one_hot_a) + (1 - lam) * nn.CrossEntropyLoss()(softmax_out_mix, mixup_one_hot_b)
                loss =(loss_3-loss_2)+loss_7
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                y_batch = model(transform(x_test_batch), cls_features=cls_features)
                acc+= (y_batch['logits'].max(1)[1] == y_test_batch).float().sum()
            acc=acc.item() / x_test.shape[0]
            print('__________________________________________________________'+corruption+str(severity)+'____________________________________________________')
            print((acc))
            result_pool[corruption+str(severity)]=acc
if __name__ == '__main__':
    main()

