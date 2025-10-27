import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

attn_maps = {}
def hook_fn(name):
    def forward_hook(module, input, output):
        if hasattr(module.processor, "attn_map"):
            attn_maps[name] = module.processor.attn_map
            del module.processor.attn_map

    return forward_hook

def register_cross_attention_hook(unet):
    for name, module in unet.named_modules():
        if name.split('.')[-1].startswith('attn2'):
        # if name.split('.')[-1].startswith('attn1'):
            module.register_forward_hook(hook_fn(name))

    return unet

def upscale(attn_map, target_size):
    attn_map = torch.mean(attn_map, dim=0)
    attn_map = attn_map.permute(1,0)
    temp_size = None

    for i in range(0,5):
        scale = 2 ** i
        if ( target_size[0] // scale ) * ( target_size[1] // scale) == attn_map.shape[1]*64:
            temp_size = (target_size[0]//(scale*8), target_size[1]//(scale*8))
            break

    assert temp_size is not None, "temp_size cannot is None"

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




# attn_maps = {}

# def hook_fn(name):
#     def forward_hook(module, input, output):
#         if hasattr(module.processor, "attn_map"):
#             attn_maps[name] = module.processor.attn_map
#             del module.processor.attn_map
#     return forward_hook

# def register_cross_attention_hook(unet):
#     for name, module in unet.named_modules():
#         if name.split('.')[-1].startswith('attn2'):
#             module.register_forward_hook(hook_fn(name))
#     return unet

# def upscale(attn_map, target_size):
#     print("attn_map形状:", attn_map.shape)
#     attn_map = torch.mean(attn_map, dim=0)
#     if attn_map.is_sparse:
#         attn_map = attn_map.to_dense()
    
#     if attn_map.dim() == 4:
#         attn_map = attn_map.permute(2, 0, 1, 3)
#     elif attn_map.dim() == 3:
#         attn_map = attn_map.permute(1, 0, 2)
    
#     temp_size = None
#     for i in range(0, 5):
#         scale = 2 ** i
#         if (target_size[0] // scale) * (target_size[1] // scale) == attn_map.shape[1] * 64:
#             temp_size = (target_size[0] // (scale * 8), target_size[1] // (scale * 8))
#             break
    
#     # 如果没有找到合适的temp_size，使用一个默认值
#     if temp_size is None:
#         temp_size = (target_size[0] // 8, target_size[1] // 8)
    
#     attn_map = attn_map.view(attn_map.shape[0], *temp_size)
#     attn_map = F.interpolate(
#         attn_map.unsqueeze(0).to(dtype=torch.float32),
#         size=target_size,
#         mode='bilinear',
#         align_corners=False
#     )[0]
    
#     attn_map = torch.softmax(attn_map, dim=0)
#     return attn_map

# def get_net_attn_map(image_size, batch_size=2, instance_or_negative=False, detach=True):
#     idx = 0 if instance_or_negative else 1
#     net_attn_maps = []

#     for name, attn_map in attn_maps.items():
#         attn_map = attn_map.cpu() if detach else attn_map
#         attn_map = torch.chunk(attn_map, batch_size)[idx].squeeze()
#         attn_map = upscale(attn_map, image_size)
#         net_attn_maps.append(attn_map)

#     net_attn_maps = torch.mean(torch.stack(net_attn_maps, dim=0), dim=0)
#     return net_attn_maps

# def attnmaps2images(net_attn_map, prompt_tokens, target_images):
#     query_tokens = len(prompt_tokens)  # Number of tokens to visualize
#     images = []

#     for i in range(query_tokens):
#         attn_map = net_attn_map[i].cpu().numpy()
#         normalized_attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map)) * 255
#         normalized_attn_map = normalized_attn_map.astype(np.uint8)
#         attention_img = Image.fromarray(normalized_attn_map)

#         # Overlap attention map onto target image for visualization
#         overlay_img = Image.blend(target_images[i], attention_img, alpha=0.5)
#         images.append(overlay_img)

#     return images


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