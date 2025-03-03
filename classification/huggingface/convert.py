import torch
from PIL import Image
from transformers import (AutoConfig, AutoModel,
                          AutoModelForImageClassification, CLIPImageProcessor)
from transformers.modeling_outputs import BackboneOutput


def convert_checkpoint(old_path, new_path):
    print(f'old_path: {old_path}, new_path: {new_path}')
    image_path = 'img_1.png'
    image_processor = CLIPImageProcessor.from_pretrained(new_path)
    image = Image.open(image_path)
    image = image_processor(images=image, return_tensors='pt').pixel_values
    print('image shape:', image.shape)

    config = AutoConfig.from_pretrained(new_path, trust_remote_code=True)
    model = AutoModelForImageClassification.from_config(config, trust_remote_code=True)

    checkpoint = torch.load(old_path)['model']
    new_checkpoint = {}
    for k, v in checkpoint.items():
        if 'gamma' in k:
            k = k.replace('gamma1', 'layer_scale1')
            k = k.replace('gamma2', 'layer_scale2')
        k = 'model.' + k
        new_checkpoint[k] = v

    checkpoint = new_checkpoint
    message = model.load_state_dict(checkpoint, strict=False)
    print(message)

    model.save_pretrained(new_path)
    print('done')

    # image = torch.rand(1, 3, 224, 224)
    output = model(image)
    for k, v in output.items():
        if type(v) == list:
            for idx, item in enumerate(v):
                print(f'{k}_{idx} shape: {item.shape}')
        elif v is None:
            continue
        else:
            print(f'{k} shape: {v.shape}')

    logits = output['logits']
    argmax = int(torch.argmax(logits, dim=1))
    print(argmax)


convert_checkpoint(old_path='pretrained/internimage_l_22k_384.pth',
                   new_path='22k_model/internimage_l_22k_384')
convert_checkpoint(old_path='pretrained/internimage_xl_22k_384.pth',
                   new_path='22k_model/internimage_xl_22k_384')
convert_checkpoint(old_path='pretrained/internimage_h_jointto22k_384.pth',
                   new_path='22k_model/internimage_h_jointto22k_384')
convert_checkpoint(old_path='pretrained/internimage_g_jointto22k_384.pth',
                   new_path='22k_model/internimage_g_jointto22k_384')
convert_checkpoint(old_path='pretrained/internimage_t_1k_224.pth',
                   new_path='in1k_model/internimage_t_1k_224')
convert_checkpoint(old_path='pretrained/internimage_s_1k_224.pth',
                   new_path='in1k_model/internimage_s_1k_224')
convert_checkpoint(old_path='pretrained/internimage_b_1k_224.pth',
                   new_path='in1k_model/internimage_b_1k_224')
convert_checkpoint(old_path='pretrained/internimage_l_22kto1k_384.pth',
                   new_path='in1k_model/internimage_l_22kto1k_384')
convert_checkpoint(old_path='pretrained/internimage_xl_22kto1k_384.pth',
                   new_path='in1k_model/internimage_xl_22kto1k_384')
convert_checkpoint(old_path='pretrained/internimage_h_22kto1k_640.pth',
                   new_path='in1k_model/internimage_h_22kto1k_640')
convert_checkpoint(old_path='pretrained/internimage_g_22kto1k_512.pth',
                   new_path='in1k_model/internimage_g_22kto1k_512')
