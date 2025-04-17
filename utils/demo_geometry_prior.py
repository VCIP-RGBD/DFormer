import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from torchvision.transforms.functional import to_pil_image
from typing import Tuple, Union
from functools import partial
# from einops import einsum
from torchvision.transforms import Resize
import cv2
import numpy as np

class GeoPrior(nn.Module):

    def __init__(self, embed_dim=128, num_heads=4, initial_value=2, heads_range=6):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value  
        self.heads_range = heads_range 
        self.num_heads = num_heads
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
        
    def generate_pos_decay(self, H: int, W: int):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        index_h = torch.arange(H).to(self.decay) #保持一個類型
        index_w = torch.arange(W).to(self.decay) #
        grid = torch.meshgrid([index_h, index_w])
        grid = torch.stack(grid, dim=-1).reshape(H*W, 2) #(H*W 2)
        mask = grid[:, None, :] - grid[None, :, :] #(H*W H*W 2)
        mask = (mask.abs()).sum(dim=-1)
        mask = mask #* self.decay[:, None, None]  #(n H*W H*W)
        return mask
    
    def generate_2d_depth_decay(self, H: int, W: int, depth_grid):
        '''
        generate 2d decay mask, the result is (HW)*(HW)
        '''
        # index_h = torch.arange(H).to(self.decay) #保持一個類型
        # index_w = torch.arange(W).to(self.decay) #
        # grid = torch.meshgrid([index_h, index_w])
        # grid = torch.stack(grid, dim=-1).reshape(H*W, 2) #(H*W 2)
        # to do: resize depth_grid to H,W
        # print(depth_grid.shape,H,W,'2d')
        B,_,H,W = depth_grid.shape
        grid_d = depth_grid.reshape(B, H*W, 1)
        print(grid_d.dtype,'aaaaaaaaaaaaaaaaaa')
        # exit()
        mask_d = grid_d[:, :, None, :] - grid_d[:, None,:, :] #(H*W H*W)
        # mask = grid[:, None, :] - grid[None, :, :] #(H*W H*W 2)
        # print(mask_d.shape, self.decay[None, :, None, None].shape,'11111')
        mask_d = (mask_d.abs()).sum(dim=-1)
        # print(torch.max(mask_d),torch.min(mask_d))
        # exit()
        mask_d = mask_d.unsqueeze(1) #* self.decay[None, :, None, None].cuda()  #(n H*W H*W)
        return mask_d
    
    
    
    def forward(self, slen: Tuple[int], depth_map, activate_recurrent=False, chunkwise_recurrent=False):
        '''
        slen: (h, w)
        h * w == l
        recurrent is not implemented
        '''
        # print(depth_map.shape,'depth_map')
        depth_map = F.interpolate(depth_map, size=slen,mode='bilinear',align_corners=False)
        # print(depth_map.shape,'downsampled')
        depth_map = depth_map.float()
        # depth_map = Resize(slen[0],slen[1])(depth_map).reshape(slen[0],slen[1])
        
        index = torch.arange(slen[0]*slen[1]).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :]) #(l d1)
        sin = sin.reshape(slen[0], slen[1], -1) #(h w d1)
        cos = torch.cos(index[:, None] * self.angle[None, :]) #(l d1)
        cos = cos.reshape(slen[0], slen[1], -1) #(h w d1)
        mask_1 = self.generate_pos_decay(slen[0], slen[1]) #(n l l)
        mask_d = self.generate_2d_depth_decay(slen[0], slen[1], depth_map)
        print(torch.max(mask_d),torch.min(mask_d),'-2')
        mask = mask_d#/torch.max(mask_d, dim=0)[0] #mask.cuda() * (2*(1-
        mask_sum = (0.85*mask_1.cuda()+0.15*mask) * self.decay[:, None, None].cuda()
        retention_rel_pos = ((sin, cos), mask, mask_1, mask_sum)
        print(mask.shape,mask_1.shape)
        # exit()

        return retention_rel_pos

def fangda(mask, in_size=(480//20,640//20), out_size=(480,640)):
    new_mask = torch.zeros(out_size)
    ratio_h, ratio_w = out_size[0]//in_size[0], out_size[1]//in_size[1]
    for i in range(in_size[0]):
        for j in range(in_size[1]):
            new_mask[i*ratio_h:(i+1)*ratio_h,j*ratio_w:(j+1)*ratio_w]=mask[i,j]
    return new_mask

def put_mask(image,mask,color_rgb=None,border_mask=False,color_temp='jet',num_c='',beta=2,fixed_num=None):
    # color_rgb eq:white-[255,255,255] num_c颜色index，同时生成多种颜色mask后挑选的话可以用这种

    # image = cv2.imread(img_path)
    # mask =cv2.imread(mask_path)
    # print(image.shape)
    # print(mask.shape)
    # mask = torch.nn.functional.interpolate(mask, size=(480,640), scale_factor=None, mode='nearest', align_corners=None)

    mask = mask.numpy()
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image,dsize=(640,480),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask,dsize=(640,480),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
    # mask=cv2.resize(mask, image.shape[1::-1], interpolation=np.INTER_NEAREST)
    print(mask.shape,image.shape, torch.max(torch.tensor(image)),torch.min(torch.tensor(image)),'0000000000')


    color=np.zeros((1,1,3), dtype=np.uint8)
    if color_rgb is not None:
        color[0,0,2],color[0,0,1],color[0,0,0]=color_rgb
    else:
        color[0, 0, 2], color[0, 0, 1], color[0, 0, 0]=120,86,87
        # # 140,86,87
    # 0, 178, 238
    # 255 ,48 ,48
    # 255 ,127, 0
    if fixed_num is not None:
        mask = ((1-mask/255))
    else:
        mask=(1-mask/np.max(mask))#*0.5+0.5
    # # mask=mask*color
    # # print(mask.shape, torch.min(mask), torch.max(mask))
    # jet_mask = mask.astype(np.uint8)#cv2.imread("nezha.jpg",cv2.IMREAD_GRAYSCALE)
    # # for i in range(22):
    # jet_mask = cv2.applyColorMap(jet_mask,2) 
    # # cv2.imshow('map',dst) 
    # # cv2.waitKey(500)
    # # cv2.imwrite("map-"+str(2)+".jpg",dst)
    # # jet_mask_np = torch.tensor(jet_mask)
    # # print(jet_mask_np.shape,'dst')

    # mask=jet_mask.astype(np.uint8)

    # # alpha 为第一张图片的透明度
    # alpha = 1
    # # beta 为第二张图片的透明度
    # beta = 0.5
    # gamma = 0
    # print(str(beta))
    print('cammmmmmm',image.size,mask.size)
    from torchcam.utils import overlay_mask
    result = overlay_mask(to_pil_image(image.astype(np.uint8)), to_pil_image(mask), colormap = color_temp, alpha=0.4)

    # mask_img = cv2.addWeighted((image*(1-beta*0.3)).astype(np.uint8), alpha, mask, beta, gamma)
    # print(os.path.join(output_fold, img_name))
    # cv2.imwrite(os.path.join(output_fold, img_name.replace('.png',num_c)+'.png'), mask_img)
    return np.array(result)#mask_img


cmap_list = ['jet_r']
# 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r']


H = 480//20
W = 640//20
file_list = [303] # which file to show
index_list = [[584]] # which index to show
from grid_gen import gen_grid_save
for i_temp in range(len(file_list)):
    file_index = file_list[i_temp]
    RGB_path = './datasets/NYUDepthv2/RGB/'+str(file_index)+'.jpg' #your path
    Depth_path = './datasets/NYUDepthv2/Depth/'+str(file_index)+'.png' #your path
    

    grid_d = cv2.imread(Depth_path,0)
    grid_d = cv2.resize(grid_d,dsize=(W,H),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
    cv2.imwrite('0_pool.png',grid_d)

    grid_d = torch.tensor(grid_d).reshape(1,1,H,W)
    grid_d_copy=cv2.imread(Depth_path)
    grid_d_copy = cv2.resize(grid_d_copy,dsize=(640,480),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
    grid_d_copy_gray = cv2.imread(Depth_path,0)
    grid_d_copy_gray = cv2.resize(grid_d_copy_gray,dsize=(640,480),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
    print('min max', torch.max(grid_d), torch.min(grid_d))
    print(grid_d.shape)
    grid_d=grid_d.cuda()
    #  接入depth

    respos = GeoPrior()
    ((sin,cos), depth_map, mask_1, mask_sum) = respos((H,W), grid_d)
    print(depth_map.shape, mask_1.shape,'-1')
    # mask_1=mask_1.transpose(-2,-1)
    print(torch.max(depth_map),torch.min(depth_map))
    # exit()

    # 
    img_path = RGB_path
    img = cv2.imread(img_path)
    img = cv2.resize(img,dsize=(640,480),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)

    grid_d_old = cv2.imread(Depth_path,0)
    grid_d_old = cv2.resize(grid_d_old,dsize=(W,H),fx=1,fy=1,interpolation=cv2.INTER_LINEAR)
    # cv2.imwrite('0_pool.png',grid_d)
    grid_d_old = torch.tensor(grid_d_old).reshape(H*W,1)
    # print('min max', torch.max(grid_d), torch.min(grid_d))
    # print(grid_d.shape)
    grid_d=grid_d.cuda()
    mask_d_old = grid_d_old[:, None, :] - grid_d_old[None, :, :] #(H*W H*W 2)
    mask_d_old = (mask_d_old.abs()).sum(dim=-1)
    # 
    # color_list = []
    # for i in range(100):
    #     color_list.append([])
    # print(depth_map.shape,'-------------00000000')
    Color_N=255
    for i in index_list[i_temp]:#range(0,H*W,4):#range(0,H*W,4):#index_list[i_temp]:#range(0,H*W,1): [242,258]:#range
        for color_temp in cmap_list:
            print(i,'index')
            # print(i,depth_map[0,0,i,:].shape)

            temp_mask_d = depth_map[0,0,i,:].reshape(H,W).cpu()

            window_attn = 255*torch.ones((H,W)).reshape(H,W).cpu()
            print(window_attn.shape,'window')
            for h in range(H):
                for w in range(W):
                    if (h>=(H//2)) & (w>=(W//2)):
                        window_attn[h,w]=Color_N
            print(torch.max(window_attn),torch.min(window_attn))

            local_attn = 255*torch.ones((H,W)).reshape(H,W).cpu()
            print(local_attn.shape,'local_attn')
            for h in range(H):
                for w in range(W):
                    if ((h>15)&(h<21)) & ((w>25)&(w<31)):
                        local_attn[h,w]=Color_N
            print(torch.max(local_attn),torch.min(local_attn))
            
            cv2.imwrite('temp.png',window_attn.cpu().numpy())
            temp_mask = mask_1[i,:].reshape(H,W).cpu()
            print(torch.max(temp_mask_d),torch.min(temp_mask_d))
            temp_mask_d_old = mask_d_old[i,:].reshape(H,W).cpu()
            temp_mask_sum = mask_sum[0,0,i,:].reshape(H,W).cpu()
            temp_mask_d=torch.nn.functional.normalize(temp_mask_d, p=2.0, dim=1, eps=1e-12, out=None)

            temp_mask_d = 255*(temp_mask_d-torch.min(temp_mask_d))/(torch.max(temp_mask_d)-torch.min(temp_mask_d))
            
            temp_mask = 255*((temp_mask-torch.min(temp_mask))/(torch.max(temp_mask)-torch.min(temp_mask)))

            temp_mask_sum = 255*((temp_mask_sum-torch.min(temp_mask_sum))/(torch.max(temp_mask_sum)-torch.min(temp_mask_sum)))
            # temp_mask[temp_mask>20]=255
            gama =0.55
            temp_mask_d_old = 255*(temp_mask_d_old-torch.min(temp_mask_d_old))/(torch.max(temp_mask_d_old)-torch.min(temp_mask_d_old))
            a0=put_mask(img,fangda(temp_mask),color_temp=color_temp)
            jiange = 255*torch.ones(img.shape[0],20)
            #  resize fangda
            temp_mask_fuse = torch.cat([fangda(temp_mask),jiange,fangda(temp_mask_d),jiange,fangda(gama*temp_mask+(1-gama)*temp_mask_d),jiange,torch.tensor(grid_d_copy_gray)],dim=1)
            # cv2.imwrite('./temp/'+str(file_index)+"_"+str(i)+'_gray.png',temp_mask_fuse.cpu().numpy())
            jiange = np.ones((img.shape[0],20, 3)) * 255
            
            a2 = put_mask(img, fangda(temp_mask_d),color_temp=color_temp)
            print(a2.shape)
            a3 = put_mask(img,fangda(gama*temp_mask+(1-gama)*temp_mask_d),color_temp=color_temp)
            
            
            
            
            image = np.concatenate([img,grid_d_copy,a0,jiange, a2,jiange,a3,jiange] ,axis=1)
            cv2.imwrite('./temp/'+str(file_index)+color_temp+"_"+str(i)+'.png',image)

