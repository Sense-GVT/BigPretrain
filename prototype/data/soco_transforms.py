import random
import numpy as np
from PIL import ImageFilter
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms.transforms import RandomResizedCrop
import torchvision.transforms.functional as TF
import springvision
from PIL import ImageDraw
import math
import datetime
import random

class RandomResizedCropAlign(RandomResizedCrop):
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        
        return TF.resized_crop(img, i, j, h, w, self.size, self.interpolation), [j, i, w, h]

class GaussianBlur_BYOL(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

def fuc_box_jitter(box, input_size, scale=0.1):
    tmp_scale = random.uniform(-scale, scale)
    x_jitter = max(box[0]+(box[2]-box[0])*tmp_scale, 0)
    tmp_scale = random.uniform(-scale, scale)
    y_jitter = max(box[1]+(box[3]-box[1])*tmp_scale, 0)
    tmp_scale = random.uniform(-scale, scale)
    w_jitter = (box[2]-box[0]) + (box[2]-box[0])*tmp_scale
    tmp_scale = random.uniform(-scale, scale)
    h_jitter = (box[3]-box[1]) + (box[3]-box[1])*tmp_scale
    return [x_jitter, y_jitter,min(x_jitter+w_jitter, input_size), min(y_jitter+h_jitter,input_size)]

class SocoTransform:
    def __init__(self, base_transform, cfg_dataset, K=4):
        self.input_size = cfg_dataset['input_size']
        if 'v3_szie' in cfg_dataset.keys():
            self.down_size = cfg_dataset['v3_size']
        else:
            self.down_size = self.input_size // pow(2, 6) * pow(2, 5)
        self.base_crop=transforms.Compose([transforms.Resize(self.input_size), transforms.CenterCrop(self.input_size)])
        self.downsample=transforms.Compose([transforms.Resize(self.down_size)])
        self.down_ratio = self.input_size/self.down_size
        self.use_box_jitter = cfg_dataset['use_box_jitter']

        if 'pad' in cfg_dataset.keys():
            self.pad = cfg_dataset['pad']
        else:
            self.pad = False
        if cfg_dataset['proposal_num'] is not None and cfg_dataset['proposal_num']!='None':
            self.K = cfg_dataset['proposal_num']
        else:
            self.K = K
        if cfg_dataset['random_assign'] is not None and cfg_dataset['random_assign']!='None':
            self.random_assign = cfg_dataset['random_assign']
        else:
            self.random_assign = False

        self.HFlip = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
        self.base_transform1 = transforms.Compose([
                            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.RandomApply([GaussianBlur_BYOL(kernel_size=int(0.1 * self.input_size))], p=1.0),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
        self.base_transform2 = transforms.Compose([
                            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.RandomApply([GaussianBlur_BYOL(kernel_size=int(0.1 * self.input_size))], p=0.1),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
        self.base_transform3 = transforms.Compose([
                            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.RandomApply([GaussianBlur_BYOL(kernel_size=int(0.1 * self.input_size//2))], p=0.1),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])

        self.crop = RandomResizedCropAlign(self.input_size, scale=(0.5, 1.))
        
    def __call__(self, x, proposal=None, test=False):
        view1 = self.base_crop(x)
        view2, crop_box= self.crop(view1)
        view3 = self.downsample(view2)
        if proposal is None:
            img_lbl, regions = selectivesearch.selective_search(np.array(view1), scale=50, sigma=0.9, min_size=10)
        else:
            img_size = x.size
            short_edge = 0 if img_size[0]<img_size[1] else 1
            resize_ratio = img_size[short_edge] / self.input_size
            ccrop_size = (img_size[short_edge-1] / resize_ratio - self.input_size)/2
            regions = []

            for ele in proposal:
                if isinstance(ele, dict):
                    ele = ele['bbox']
                x1,y1,x2,y2=ele[0]/resize_ratio, ele[1]/resize_ratio, ele[2]/resize_ratio, ele[3]/resize_ratio
                if short_edge:
                    x1 = max(x1-ccrop_size,0)
                    x2 = min(x2-ccrop_size,self.input_size)
                else:
                    y1 = max(y1-ccrop_size,0)
                    y2 = min(y2-ccrop_size,self.input_size)
                
                regions.append([x1,y1,x2-x1,y2-y1])
        view1_regions,view2_regions,view3_regions = [],[],[]
        view1_regions_optional,view2_regions_optional,view3_regions_optional = [],[],[]

        for region in regions:
            if proposal is None:
                rect = region['rect']
            else:
                rect = region

            if self.use_box_jitter and random.random()>0.5:
                tmp_scale = random.uniform(-0.1,+0.1)
                w_jitter = rect[2]*tmp_scale
                h_jitter = rect[3]*tmp_scale
                x_jitter = max(rect[0]+w_jitter,0)
                y_jitter = max(rect[0]+h_jitter,0)
                w_jitter += rect[2]
                h_jitter += rect[3]
                x, y, w, h = x_jitter, y_jitter, w_jitter, h_jitter
            else:
                x, y, w, h = rect
            x2 = min(x+w, self.input_size)
            y2 = min(y+h, self.input_size)
            xa = (max(x-crop_box[0], 0))*self.input_size/crop_box[2]
            ya = (max(y-crop_box[1],0))*self.input_size/crop_box[3]
            wa = w*self.input_size/crop_box[2]
            ha = h*self.input_size/crop_box[3]
            x2a = min(xa+wa,self.input_size)
            y2a = min(ya+ha,self.input_size)
            
            view1_regions_optional.append([x, y, x2, y2])
            view2_regions_optional.append([xa, ya, x2a, y2a])
            view3_regions_optional.append([xa/self.down_ratio, ya/self.down_ratio, x2a/self.down_ratio, y2a/self.down_ratio])
            
            if not test and h!=0 and float(w/h)<=3 and float(w/h)>=1/3 and math.sqrt(w*h) <= 0.8*self.input_size \
                    and math.sqrt(w*h)>=0.3*self.input_size:
                view1_regions.append([x, y, x2, y2])
                view2_regions.append([xa, ya, x2a, y2a])
                view3_regions.append([xa/self.down_ratio, ya/self.down_ratio, (xa+wa)/self.down_ratio, (ya+ha)/self.down_ratio])

        if len(view1_regions) >= self.K:
            sample_idx = list(range(len(view1_regions)))
            sample_idx = random.sample(sample_idx, self.K)
            view1_regions_gt = [view1_regions[s] for s in sample_idx]

            view1_regions = [view1_regions[s] for s in sample_idx]
            
            view2_regions = [view2_regions[s] for s in sample_idx]
            view3_regions = [view3_regions[s] for s in sample_idx]
        else:
            view1_regions_gt = view1_regions.copy()
            if len(view1_regions) > 0:
                view1_regions_optional = view1_regions
            else:
                pass

            while len(view1_regions) < self.K :
                sample_idxs = list(range(len(view1_regions_optional)))
                sample_idx = random.sample(sample_idxs, 1)
                tmp = view1_regions_optional[sample_idx[0]]
                tmp_scale = random.uniform(-0.1, 0.1)
                w_jitter = (tmp[2]-tmp[0])*tmp_scale
                h_jitter = (tmp[3]-tmp[1])*tmp_scale
                x_jitter = max(tmp[0]+w_jitter,0)
                y_jitter = max(tmp[1]+h_jitter,0)
                w_jitter += (tmp[2]-tmp[0])
                h_jitter += (tmp[3]-tmp[1])

                xa = (max(x_jitter-crop_box[0],0))*self.input_size/crop_box[2]
                ya = (max(y_jitter-crop_box[1],0))*self.input_size/crop_box[3]
                wa = w_jitter*self.input_size/crop_box[2]
                ha = h_jitter*self.input_size/crop_box[3]
                x2a = min(xa+wa,self.input_size)
                y2a = min(ya+ha,self.input_size)
                view1_regions += [[x_jitter, y_jitter,min(x_jitter+w_jitter,self.input_size), min(y_jitter+h_jitter,self.input_size)]]
                view2_regions += [[xa, ya, x2a,y2a]]
                view3_regions += [[xa/self.down_ratio, ya/self.down_ratio, x2a/self.down_ratio, y2a/self.down_ratio]]
                view1_regions_gt += [tmp]
            
        if random.random() > 0.5:
            view1 = self.HFlip(view1)
            view1_regions = [[self.input_size-reg[2],reg[1],self.input_size-reg[0],reg[3]] for reg in view1_regions]
            view1_regions_gt = [[self.input_size-reg[2],reg[1],self.input_size-reg[0],reg[3]] for reg in view1_regions_gt]
        if random.random() > 0.5:
            view2 = self.HFlip(view2)
            view2_regions = [[self.input_size-reg[2],reg[1],self.input_size-reg[0],reg[3]] for reg in view2_regions]
        if random.random() > 0.5:
            view3 = self.HFlip(view3)
            view3_regions = [[self.input_size-reg[2],reg[1],self.input_size-reg[0],reg[3]] for reg in view3_regions]

        view1 = self.base_transform1(view1)
        view2 = self.base_transform2(view2)
        view3 = self.base_transform3(view3)

        fpn_level_list0,fpn_level_list1,fpn_level_list2 = [],[],[]
        fpn_level_lists = [fpn_level_list0,fpn_level_list1,fpn_level_list2 ]
        
        for k, view_regions in enumerate([view1_regions, view2_regions, view3_regions]):
            level_view = []
            for ele in view_regions:
                # length = min(ele[2]-ele[0], ele[3]-ele[1])   #min length
                area = (ele[2]-ele[0]) * (ele[3]-ele[1])
                if area < 48* 48 * (self.input_size//224):
                    fpn_level = 0
                elif area < 96 * 96 *(self.input_size//224):
                    fpn_level = 1
                elif area < 192 * 192 *(self.input_size//224):
                    fpn_level = 2
                else:
                    fpn_level = 3
                level_view.append(fpn_level)
            fpn_level_lists[k].append(level_view)

        assert len(view1_regions)==self.K, print(f'view1 need {self.K} proposal', f'but only get {len(view1_regions)}') 
        assert len(view2_regions)==self.K, print(f'view2 need {self.K} proposal', f'but only get {len(view2_regions)}')
        assert len(view3_regions)==self.K, print(f'view3 need {self.K} proposal', f'but only get {len(view3_regions)}')
        
        return [view1, view2, view3], [view1_regions, view2_regions, view3_regions], fpn_level_lists

class ToGrayscale(object):
    """Convert image to grayscale version of image."""

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        return TF.to_grayscale(img, self.num_output_channels)


class AdjustGamma(object):
    """Perform gamma correction on an image."""

    def __init__(self, gamma, gain=1):
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img):
        return TF.adjust_gamma(img, self.gamma, self.gain)


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return torch.cat([q, k], dim=0)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Cutout(object):
    """Randomly mask out one or more patches from an image."""

    def __init__(self, n_holes=2, length=32, prob=0.5):
        self.n_holes = n_holes
        self.length = length
        self.prob = prob

    def __call__(self, img):
        if np.random.rand() < self.prob:
            h = img.size(1)
            w = img.size(2)
            mask = np.ones((h, w), np.float32)
            for n in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)
                mask[y1:y2, x1:x2] = 0.

            mask = torch.from_numpy(mask)
            mask = mask.expand_as(img)
            img = img * mask

        return img


class RandomOrientationRotation(object):
    """Randomly select angles for rotation."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = random.choice(self.angles)
        return TF.rotate(img, angle)


torch_transforms_info_dict = {
    'resize': transforms.Resize,
    'center_crop': transforms.CenterCrop,
    'random_resized_crop': transforms.RandomResizedCrop,
    'random_horizontal_flip': transforms.RandomHorizontalFlip,
    'ramdom_vertical_flip': transforms.RandomVerticalFlip,
    'random_rotation': transforms.RandomRotation,
    'color_jitter': transforms.ColorJitter,
    'normalize': transforms.Normalize,
    'to_tensor': transforms.ToTensor,
    'adjust_gamma': AdjustGamma,
    'to_grayscale': ToGrayscale,
    'cutout': Cutout,
    'random_orientation_rotation': RandomOrientationRotation,
    'gaussian_blur': GaussianBlur,
    'compose': transforms.Compose
}

kestrel_transforms_info_dict = {
    'resize': springvision.Resize,
    'random_resized_crop': springvision.RandomResizedCrop,
    'random_crop': springvision.RandomCrop,
    'center_crop': springvision.CenterCrop,
    'color_jitter': springvision.ColorJitter,
    'normalize': springvision.Normalize,
    'to_tensor': springvision.ToTensor,
    'adjust_gamma': springvision.AdjustGamma,
    'to_grayscale': springvision.ToGrayscale,
    'compose': springvision.Compose,
    'random_horizontal_flip': springvision.RandomHorizontalFlip
}


def build_transformer(cfgs, image_reader={}):
    transform_list = []
    image_reader_type = image_reader.get('type', 'pil')
    if image_reader_type == 'pil':
        transforms_info_dict = torch_transforms_info_dict
    else:
        transforms_info_dict = kestrel_transforms_info_dict
        if image_reader.get('use_gpu', False):
            springvision.KestrelDevice.bind('cuda',
                                            torch.cuda.current_device())

    for cfg in cfgs:
        transform_type = transforms_info_dict[cfg['type']]
        kwargs = cfg['kwargs'] if 'kwargs' in cfg else {}
        transform = transform_type(**kwargs)
        transform_list.append(transform)
    return transforms_info_dict['compose'](transform_list)
