# PAINT:Dynamic Prompt Allocation and Tuning for Continual Test-Time Adaptation #
The code of Dynamic Prompt Allocation and Tuning for Continual Test-Time Adaptation


Abstract Continual test-time adaptation (CTTA) has recently emerged
to adapt a pre-trained source model to continuously evolving target distributions, which accommodates the dynamic nature of real-world environments. To address the potential issue of catastrophic forgetting in CTTA, existing methods typically incorporate explicit regularization terms to constrain the variation of model parameters. However, they cannot fundamentally resolve catastrophic forgetting because they rely on a single shared model to adapt across all target domains, which inevitably leads to severe inter-domain interference. In this paper, we introduce learnable domain-specific prompts that guide the model to adapt to corresponding target domains, thereby partially disentangling the parameter space of different domains. In the absence of domain identity for target samples, we propose a novel dynamic Prompt AllocatIon aNd Tuning (PAINT) method, which utilizes a query mechanism to dynamically determine whether the current samples come from a known domain or an unexplored one. For known domains, the corresponding domain-specific prompt is directly selected, while for previously unseen domains, a new prompt is allocated. Prompt tuning is subsequently performed using mutual information maximization along with structural regularization. Extensive experiments on three benchmark datasets demonstrate the effectiveness of our PAINT method for CTTA. We have released our code at https://github.com/XXX.

On the following tasks
+ CIFAR10 -> CIFAR10C (standard/gradual/shuffle)
+ ImageNet -> ImageNetC(standard)
+ ImageNet -> ImageNetR(standard)

Compare this with the following methods
+ CoTTA
+ ETA
+ CPL
+ RMT
+ SAR
+ PAINT

## Install ##
```git clone https://github.com/Cadezzyr/PAINT.git```  
## Test on CIFAR10 -> CIFAR10C tasks (standard/gradual/shuffle) ##
```
cd cifar
python test_cifar10.py
python test_cifar10_gradual.py
python test_cifar10_shuffle.py
```
## Test on ImageNet -> ImageNetC tasks (standard) ##
```
cd imagenet
python test_imagenet.py
```
## Test on ImageNet -> ImageNetR tasks (standard) ##
```
cd imagenet
python test_imagenet-r-200classes.py
```

