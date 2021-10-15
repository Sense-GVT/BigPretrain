import torch
from torchvision import models
import os
import argparse
import re

parser = argparse.ArgumentParser(description='')
parser.add_argument('checkpoint', type=str)
parser.add_argument('--fpn_head', action='store_true')
parser.add_argument('--model', default='mr50')
args = parser.parse_args()


def trans2r50(ch_path,save_path):
    if torch.cuda.is_available():
        pretrain = torch.load(ch_path)
    else:
        pretrain = torch.load(ch_path, map_location=torch.device('cpu'))
    
    tmp_dict = {}
    pretrain_dict = pretrain['model']
    for key, val in pretrain_dict.items():
        old_key = key
        if 'module.encoder_q.backbone' in key:
            drop_prex = 3 
            key = '.'.join(key.split('.')[drop_prex:])
            if "layer" not in key:
                key = "stem." + key
            for t in [1, 2, 3, 4]:
                key = key.replace("layer{}".format(t), "res{}".format(t + 1))
            for t in [1, 2, 3]:
                key = key.replace("bn{}".format(t), "conv{}.norm".format(t))
            if 'downsample.0' in key or 'downsample.1' in key:
                key = key.replace("downsample.0", "shortcut")
                key = key.replace("downsample.1", "shortcut.norm")
            else:
                key = key.replace("downsample", "shortcut")
            key = 'backbone.bottom_up.'+key
            tmp_dict[key] = val
            print(old_key, "->", key)
        if args.fpn_head:
            flag = False
            if 'module.encoder_q.head.fc6' in key:
                key = key.replace('module.encoder_q.head.fc6', 'roi_heads.box_head.fc1')
                flag = True
            if 'module.encoder_q.head.fc7' in key:
                key = key.replace('module.encoder_q.head.fc7', 'roi_heads.box_head.fc2')
                flag = True
            if 'module.encoder_q.fpn' in key:
                level, index = re.findall(r"\d+\.?\d*", key.split('.')[3])
                op = key.split('.')[-1]
                style = 'lateral' if index=='1' else 'output'
                key = 'backbone.fpn_'+style+str(level)+'.'+op
                flag = True
            if flag:
                tmp_dict[key] = val
                print(old_key, "->", key)

    torch.save({"model":tmp_dict}, save_path)


def get_all_weights(files,sPath):
    for schild in os.listdir(sPath):
        sChildPath = os.path.join(sPath, schild)
        if os.path.isdir(sChildPath):
            files.extend(get_all_weights([],sChildPath))
        else:
            files.append(sChildPath)
    return files

save_path = './'
print('start transfer')
print('use fpn_head:{}'.format(args.fpn_head))
file_ = [args.checkpoint ]

for ele in file_:
    print('transfering ',ele)
    sp = ele.split('/')[-1]
    if args.fpn_head:
        sp = sp.split('.')[0] + '_fpn_head.pth.tar'
    # sp = sp.split('/')[-1]
    tmp_save = save_path
    if not os.path.exists(tmp_save):
        os.makedirs(tmp_save)
    trans2r50(ele, tmp_save+sp )
