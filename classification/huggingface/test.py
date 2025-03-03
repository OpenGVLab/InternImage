import torch
from PIL import Image
from transformers import (AutoConfig, AutoModel,
                          AutoModelForImageClassification, CLIPImageProcessor)


def test_model(model_name):
    print('model_name:', model_name)
    image_path = 'img_1.png'
    image_processor = CLIPImageProcessor.from_pretrained(model_name)
    image = Image.open(image_path)
    image = image_processor(images=image, return_tensors='pt').pixel_values
    print('image shape:', image.shape)

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    output = model(image)
    for k, v in output.items():
        if type(v) == list:
            for idx, item in enumerate(v):
                print(f'{k}_{idx} shape: {item.shape}')
        elif v is None:
            print(f'{k} is None')
        else:
            print(f'{k} shape: {v.shape}')

    print('------------------------')

    model = AutoModelForImageClassification.from_pretrained(model_name, trust_remote_code=True)
    output = model(image)
    for k, v in output.items():
        if type(v) == list:
            for idx, item in enumerate(v):
                print(f'{k}_{idx} shape: {item.shape}')
        elif v is None:
            print(f'{k} is None')
        else:
            print(f'{k} shape: {v.shape}')
    logits = output['logits']
    argmax = int(torch.argmax(logits, dim=1))
    print(argmax)


test_model('./22k_model/internimage_l_22k_384')
test_model('./22k_model/internimage_xl_22k_384')
test_model('./22k_model/internimage_h_jointto22k_384')
test_model('./22k_model/internimage_g_jointto22k_384')
test_model('./in1k_model/internimage_t_1k_224')
test_model('./in1k_model/internimage_s_1k_224')
test_model('./in1k_model/internimage_b_1k_224')
test_model('./in1k_model/internimage_l_22kto1k_384')
test_model('./in1k_model/internimage_xl_22kto1k_384')
test_model('./in1k_model/internimage_h_22kto1k_640')
test_model('./in1k_model/internimage_g_22kto1k_512')
