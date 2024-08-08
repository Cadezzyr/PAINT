# PAINT
The code of Dynamic Prompt Allocation and Tuning for Continual Test-Time Adaptation


Abstract Continual test-time adaptation (CTTA) has recently emerged
to adapt a pre-trained source model to continuously evolving
target distributions, which accommodates the dynamic nature of real-world environments. To address the potential issue of catastrophic forgetting in CTTA, existing methods typically incorporate explicit regularization terms to constrain
the variation of model parameters. However, they cannot
fundamentally resolve catastrophic forgetting because they
rely on a single shared model to adapt across all target domains, which inevitably leads to severe inter-domain interference. In this paper, we introduce learnable domain-specific
prompts that guide the model to adapt to corresponding target domains, thereby partially disentangling the parameter
space of different domains. In the absence of domain identity for target samples, we propose a novel dynamic Prompt
AllocatIon aNd Tuning (PAINT) method, which utilizes a
query mechanism to dynamically determine whether the current samples come from a known domain or an unexplored
one. For known domains, the corresponding domain-specific
prompt is directly selected, while for previously unseen domains, a new prompt is allocated. Prompt tuning is subsequently performed using mutual information maximization
along with structural regularization. Extensive experiments
on three benchmark datasets demonstrate the effectiveness of
our PAINT method for CTTA. We have released our code at
https://github.com/XXX.
