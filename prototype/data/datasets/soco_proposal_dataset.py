# -*- coding: utf-8 -*-
import torch
import json
import os.path as osp
import os
import requests
import numpy as np
import time
from typing import List
from .base_dataset import BaseDataset
from prototype.data.image_reader import build_image_reader
import linklink as link
import random
import datetime
import io
from PIL import Image

class SOCOProposalDataset(BaseDataset):

    def __init__(self, root_dir, meta_file, transform=None,
                 read_from='mc', evaluator=None, image_reader_type='pil',
                 server_cfg={}, fseek=False, soco_type=False,label_texts_ensemble='none',
                 rank_dataset=False, test=False,json_dir=None):

        if not isinstance(meta_file, List):
            meta_file = [meta_file]
        if not isinstance(root_dir, List):
            root_dir = [root_dir]
        self.soco_type = soco_type
        self.meta_file = meta_file
        self.json_dir = json_dir
        self.root_dir = root_dir
        self.read_from = read_from
        self.transform = transform
        self.evaluator = evaluator
        self.image_reader = build_image_reader(image_reader_type)
        self.server_cfg = server_cfg
        self.rank_dataset = rank_dataset
        self.fseek = fseek
        self.test = test
        self.initialized = False
        self.label_texts_ensemble = label_texts_ensemble
        self.num = 0

        self.metas = []

        for rd, each_meta_file in zip(root_dir, meta_file):
            if self.json_dir is None or self.json_dir=='None':
                with open(each_meta_file) as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.replace('\x00','')
                    try:
                        info = json.loads(line)
                    except:
                        continue
                        # print(line)
                        # set_trace()
                    info['filename'] = osp.join(rd, info['filename'])
                    # if len(info['instances']) > 8:
                    #     sample_idx = list(range(len(info['instances'])))
                    #     sample_idx = random.sample(sample_idx, 8)
                    #     info['instances'] = [info['instances'][s] for s in sample_idx]
                    
                    if len(info['instances']) > 0:
                        info['instances'] = str(info['instances'])
                        self.metas.append(info)
            else:
                self.metas = []
                for dir_ in self.json_dir:
                    self.metas.extend(os.listdir(dir_))
                # self.metas = self.metas[:2000]
            self.num += len(self.metas)
            if self.read_from == 'petrel':
                self.s3client = Client('/mnt/lustre/liufenggang/petreloss.conf')
        
        super(SOCOProposalDataset, self).__init__(root_dir=root_dir,
                                          meta_file=meta_file,
                                          read_from=read_from,
                                          transform=transform,
                                          evaluator=evaluator)

    def __len__(self):
        if self.test:
            return 100
        return self.num

    def _str2list(self, x):
        if type(x) is list:
            return x
        elif type(x) is str:
            return [x]
        else:
            raise RuntimeError(
                "unknown value for _str2list: {}".format(type(x)))

    def get_data(self, idx):
        if self.json_dir is None or self.json_dir=='None':
            # try:
            curr_meta = self.metas[idx]
            proposal = eval(curr_meta['instances'])
        else:
            tmp_id = 0
            while not os.path.exists(self.json_dir[tmp_id]+self.metas[idx]):
                tmp_id += 1
                if tmp_id > len(self.json_dir):
                    assert False
            with open(self.json_dir[tmp_id]+self.metas[idx]) as f:
                lines = f.readlines()
            curr_meta = json.loads(lines[0])
            curr_meta['filename'] = osp.join(self.root_dir[0],curr_meta['filename']) 
            proposal = curr_meta['instances']
        filename = curr_meta['filename']
        # add root_dir to filename
        if self.read_from == 'mc':
            img_bytes = self.read_file(curr_meta)
            img = self.image_reader(img_bytes, filename)
        elif self.read_from == 'petrel':
            value=self.s3client.Get(filename)
            if not value:
                print('ERROR not value:',filename, flush=True)
                assert False
            img=np.fromstring(value,np.uint8)
            buff=io.BytesIO(img)
            try:
                with Image.open(buff) as img:
                    img=img.convert('RGB')
            except IOError:
                print('READ ERROR:',filename, flush=True)
                assert False

            img = np.array(img)
            if len(img.shape) != 3:
                img = np.expand_dims(img, 2).repeat(3, axis=2)
            img = Image.fromarray(img.astype('uint8')).convert('RGB')

        else:
            assert False, print('ERROR read_from')

        return curr_meta, filename, img, proposal

    def __getitem__(self, idx):
        curr_meta, filename, img, proposal = self.get_data(idx)
        while len(proposal) == 0:
            print('No Proposal:', filename, flush=True)
            idx = random.randint(0, self.num)
            curr_meta, filename, img, proposal = self.get_data(idx)
        if self.transform is not None:
            if self.soco_type=='soco':
                img, bbox, fpn_level = self.transform(img, proposal, self.test)
            elif self.soco_type=='socorpn':    
                img, img_rpn, crop_box = self.transform(img)
            else:
                img = self.transform(img)

        item = {
            'view1': img[0],
            'view2': img[1],
            'view3': img[2],
            'image_id': idx,
            'filename': filename
        }

        if self.soco_type=='soco':
            item['proposal_bbox']=bbox
            item['fpn_level']=fpn_level
        elif self.soco_type=='socorpn':
            item['img_rpn']=img_rpn
            item['crop_box']=crop_box

        return item
        # except Exception as e:
        #     print('[SOCO Proposal Dataset Exception %d]'%idx, repr(e))
        #     return self.__getitem__(random.randint(0, self.__len__()-1))

    def is_contains_chinese(self, strs):
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False

    def _get_label_text(self, text):
        label_text = ['a photo of ' + text + '.']
        if self.label_texts_ensemble == 'none':
            return label_text
        elif self.label_texts_ensemble == 'simple':
            label_text.extend(['a photo of small' + text + '.',
                               'a photo of big ' + text + '.',
                               'a picture of small' + text + '.',
                               'a picture of big' + text + '.',
                               ])
            return label_text
        else:
            return label_text

    def dump(self, writer, output):
        filenames = output['filenames']
        image_ids = output['image_ids']
        label_names = output['label_names']
        captions = output['captions']
        tags = output['tags']
        prediction = self.tensor2numpy(output['prediction'])
        score = self.tensor2numpy(output['score'])
        labels = self.tensor2numpy(output['labels'])
        for _idx in range(len(filenames)):
            res = {
                'image_id': int(image_ids[_idx]),
                'filename': filenames[_idx],
                'label': int(labels[_idx]),
                'label_name': label_names[_idx],
                'caption': captions[_idx],
                'tag': tags[_idx],
                'prediction': int(prediction[_idx]),
                'score': [float('%.8f' % s) for s in score[_idx]]
            }
            writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        writer.flush()


def py_cpu_nms(dets, thresh=0.9):
    #首先数据赋值和计算对应矩形框的面积
    #dets的数据格式是dets[[xmin,ymin,xmax,ymax,scores]....]
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    areas = (y2-y1+1) * (x2-x1+1)
    scores = dets[:,4]
    # print('areas  ',areas)
    # print('scores ',scores)
 
    #这边的keep用于存放，NMS后剩余的方框
    keep = []
    
    #取出分数从大到小排列的索引。.argsort()是从小到大排列，[::-1]是列表头和尾颠倒一下。
    index = scores.argsort()[::-1]
    # print(index) 
    #上面这两句比如分数[0.72 0.8  0.92 0.72 0.81 0.9 ]    
    #  对应的索引index[  2   5    4     1    3   0  ]记住是取出索引，scores列表没变。
    
    #index会剔除遍历过的方框，和合并过的方框。 
    while index.size >0:
        print(index.size)
        #取出第一个方框进行和其他方框比对，看有没有可以合并的
        i = index[0]       # every time the first is the biggst, and add it directly
        
        #因为我们这边分数已经按从大到小排列了。
        #所以如果有合并存在，也是保留分数最高的这个，也就是我们现在那个这个
        #keep保留的是索引值，不是具体的分数。     
        keep.append(i)
        print(keep)
        print('x1',x1[i])
        print(x1[index[1:]])
 
        #计算交集的左上角和右下角
        #这里要注意，比如x1[i]这个方框的左上角x和所有其他的方框的左上角x的
        x11 = np.maximum(x1[i], x1[index[1:]])    # calculate the points of overlap 
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        
        print(x11,y11,x22,y22)
        #这边要注意，如果两个方框相交，X22-X11和Y22-Y11是正的。
        #如果两个方框不相交，X22-X11和Y22-Y11是负的，我们把不相交的W和H设为0.
        w = np.maximum(0, x22-x11+1)    
        h = np.maximum(0, y22-y11+1)    
       
        #计算重叠面积就是上面说的交集面积。不相交因为W和H都是0，所以不相交面积为0
        overlaps = w*h
        print('overlaps is',overlaps)
        
        #这个就是IOU公式（交并比）。
        #得出来的ious是一个列表，里面拥有当前方框和其他所有方框的IOU结果。
        ious = overlaps / (areas[i]+areas[index[1:]] - overlaps)
        print('ious is',ious)
        
        #接下来是合并重叠度最大的方框，也就是合并ious中值大于thresh的方框
        #我们合并的操作就是把他们剔除，因为我们合并这些方框只保留下分数最高的。
        #我们经过排序当前我们操作的方框就是分数最高的，所以我们剔除其他和当前重叠度最高的方框
        #这里np.where(ious<=thresh)[0]是一个固定写法。
        idx = np.where(ious<=thresh)[0]
        print(idx)
 
        #把留下来框在进行NMS操作
        #这边留下的框是去除当前操作的框，和当前操作的框重叠度大于thresh的框
        #每一次都会先去除当前操作框，所以索引的列表就会向前移动移位，要还原就+1，向后移动一位
        index = index[idx+1]   # because index start from 1
        print(index)
    return keep
