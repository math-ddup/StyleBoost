import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

attn_maps = {}
def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            # 检查模块是否有一个名为 attn_map 的属性
            attn_maps[name] = module.processor.attn_map
            # 如果有，它将这个属性保存到 attn_maps 中
            del module.processor.attn_map
            # 从模块中删除这个属性，以避免内存泄漏或不必要的状态保留。

    return forward_hook

def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if name.split('.')[-1].startswith('attn2'):
            # 对于每个模块，它检查模块名称的最后一个部分是否以 'attn2' 开头
            module.register_forward_hook(hook_fn(name))

    return unet

def upscale(attn_map, target_size):
    # 将注意力图上采样到指定的目标尺寸
    attn_map = torch.mean(attn_map, dim=0)  # 沿着第一个维度对attn_map进行平均池化
    attn_map = attn_map.permute(1,0)  # 维度转换
    temp_size = None  # 初始化

    for i in range(0,5):
        scale = 2 ** i
        if ( target_size[0] // scale ) * ( target_size[1] // scale) == attn_map.shape[1]*64:
            temp_size = (target_size[0]//(scale*8), target_size[1]//(scale*8))
            break
            # 循环尝试找到一个缩放比例，使得attn_map尺寸能够与目标尺寸相匹配。

    assert temp_size is not None, "temp_size cannot is None"  # 如果没有找到，程序将抛出异常

    attn_map = attn_map.view(attn_map.shape[0], *temp_size)

    attn_map = F.interpolate(
        attn_map.unsqueeze(0).to(dtype=torch.float32),
        size=target_size,
        mode='bilinear',
        align_corners=False
    )[0]

    attn_map = torch.softmax(attn_map, dim=0)
    return attn_map
def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):

    idx = 0 if instance_or_negative else 1
    net_attn_maps = []

    for name, attn_map in attn_maps.items():
        attn_map = attn_map.cpu() if detach else attn_map
        attn_map = torch.chunk(attn_map, batch_size)[idx].squeeze()
        attn_map = upscale(attn_map, image_size) 
        net_attn_maps.append(attn_map) 

    net_attn_maps = torch.mean(torch.stack(net_attn_maps,dim=0),dim=0)

    return net_attn_maps

def attnmaps2images(net_attn_maps):

    #total_attn_scores = 0
    images = []

    for attn_map in net_attn_maps:
        attn_map = attn_map.cpu().numpy()
        #total_attn_scores += attn_map.mean().item()

        normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
        normalized_attn_map = normalized_attn_map.astype(np.uint8)
        #print("norm: ", normalized_attn_map.shape)
        image = Image.fromarray(normalized_attn_map)

        #image = fix_save_attn_map(attn_map)
        images.append(image)

    #print(total_attn_scores)
    return images
def is_torch2_available():
    return hasattr(F, "scaled_dot_product_attention")

def get_generator(seed, device):

    if seed is not None:
        if isinstance(seed, list):
            generator = [torch.Generator(device).manual_seed(seed_item) for seed_item in seed]
        else:
            generator = torch.Generator(device).manual_seed(seed)
    else:
        generator = None

    return generator
