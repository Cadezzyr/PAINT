from robustbench.data import load_cifar10c
import torch
import random

corruptions_1 = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
               'zoom_blur', 'snow','frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
# corruptions_2= ['frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
# corruptions = ['fog']
severitys = [5]
x_shuffle = None
y_shuffle = None
for corruption in corruptions_1:
    for severity in severitys:
        x_test, y_test = load_cifar10c(n_examples=10000, corruptions=[corruption], severity=severity,data_dir='/home/yrzhen/cotta-main/cifar/data', shuffle=False)
        if x_shuffle == None:
            x_shuffle = x_test
            y_shuffle = y_test
        else:
            print(x_shuffle.shape, x_test.shape)
            x_shuffle = torch.cat((x_shuffle, x_test), dim=0)
            y_shuffle = torch.cat((y_shuffle, y_test), dim=0)
new_list = []
len_all = len(x_shuffle)/50
len_all = int(len_all)
for i in range(len_all):
    new_list.append(i)

    # print(len(x_shuffle))
    # print(range(len_all))
random.shuffle(new_list)
x_test_shuffle_all = None
y_test_shuffle_all = None
for index in new_list:
    print(index)
    x_test_shuffle_once = x_shuffle[index * 50:(index + 1) * 50]
    y_test_shuffle_once = y_shuffle[index * 50:(index + 1) * 50]
    if x_test_shuffle_all == None:
        x_test_shuffle_all = x_test_shuffle_once
        y_test_shuffle_all = y_test_shuffle_once
    else:
        x_test_shuffle_all = torch.concat([x_test_shuffle_all, x_test_shuffle_once], dim=0)
        y_test_shuffle_all = torch.concat([y_test_shuffle_all, y_test_shuffle_once], dim=0)
print(x_test_shuffle_all.shape,y_test_shuffle_all.shape)
print(new_list)
#
torch.save(x_test_shuffle_all.to(torch.device('cpu')), "/media/shared_space/yrzhen/data/x_shuffle_cifar10.pth")
torch.save(y_test_shuffle_all.to(torch.device('cpu')), "/media/shared_space/yrzhen/data/y_shuffle_cifar10.pth")
x = torch.load("/media/shared_space/yrzhen/data/x_shuffle_cifar10.pth")
print(x.shape)