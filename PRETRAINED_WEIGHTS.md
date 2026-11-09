# Pretrained weights

ViT training downloads the public DeiT-S checkpoint specified by `--finetune`.
The default is Meta's public DeiT checkpoint. A local checkpoint containing
either a raw state dictionary or a dictionary with a `model` key can also be
passed to `--finetune`.

WideResNet requires the ImageNet-pretrained ResNet-38d MXNet checkpoint:

```text
pretrained/resnet_38d.params
```

Download the reuploaded checkpoint linked by the
[KAIST-vilab/AEFT repository](https://github.com/KAIST-vilab/AEFT), then install
`requirements-wideresnet.txt`. WideResNet converts the MXNet parameters at
startup.

Pretrained weights are not tracked by Git. Hugging Face VLM entry points
download their base models through `transformers` and use the user's local
Hugging Face cache.
