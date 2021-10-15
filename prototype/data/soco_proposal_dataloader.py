import random
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from .datasets import SOCOProposalDataset
from .soco_transforms import build_transformer, SocoTransform
from .sampler import build_sampler
from .metrics import build_evaluator
from .pipelines import ImageNetTrainPipeV2, ImageNetValPipeV2
from .nvidia_dali_dataloader import DaliDataloader
from easydict import EasyDict


def _collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    image_ids = [_['image_id'] for _ in batch]
    filenames = [_['filename'] for _ in batch]
    view1 = torch.stack([_['view1'] for _ in batch])
    view2 = torch.stack([_['view2'] for _ in batch])
    view3 = torch.stack([_['view3'] for _ in batch])

    output = EasyDict({
        'image_ids': image_ids,
        'filenames': filenames,
        'view1': view1,
        'view2': view2,
        'view3': view3,
    })
    if len(batch) >0 and 'proposal_bbox' in batch[0]:
        proposal_bbox1 = [torch.as_tensor(b['proposal_bbox'][0], dtype=torch.float) for b in batch]
        proposal_bbox2 = [torch.as_tensor(b['proposal_bbox'][1], dtype=torch.float) for b in batch]
        proposal_bbox3 = [torch.as_tensor(b['proposal_bbox'][2], dtype=torch.float) for b in batch]
        output['proposal_bbox1'] = proposal_bbox1
        output['proposal_bbox2'] = proposal_bbox2
        output['proposal_bbox3'] = proposal_bbox3
        output['fpn_level1'] = [b['fpn_level'][0] for b in batch]
        output['fpn_level2'] = [b['fpn_level'][1] for b in batch]
        output['fpn_level3'] = [b['fpn_level'][2] for b in batch]
        if len(batch[0]['proposal_bbox']) == 4:
            output['proposal_bbox1_gt'] = [b['proposal_bbox'][3] for b in batch]

    if len(batch) >0 and 'img_rpn' in batch[0]:   
        output['imgs_rpn'] = torch.stack([_['img_rpn'] for _ in batch])
    if len(batch) >0 and 'crop_box' in batch[0]:     
        output['crop_boxes'] = torch.as_tensor([_['crop_box'] for _ in batch], dtype=view1.dtype)
        
    return output

def build_common_augmentation(aug_type, cfg_dataset):
    """
    common augmentation settings for training/testing soco Proposal
    """
    if aug_type in ['SOCO']:
        return SocoTransform(None, cfg_dataset) 
    else:
        assert False


def build_soco_train_dataloader(cfg_dataset, data_type='train'):
    """
    build training dataloader for soco Proposal
    """
    cfg_train = cfg_dataset['train']
    # build dataset
    if cfg_dataset['use_dali']:
        # NVIDIA dali preprocessing
        assert cfg_train['transforms']['type'] == 'STANDARD', 'only support standard augmentation'
        dataset = SOCOProposalDataset(
            root_dir=cfg_train['root_dir'],
            meta_file=cfg_train['meta_file'],
            read_from=cfg_dataset['read_from'],
        )
    else:
        image_reader = cfg_dataset[data_type].get('image_reader', {})
        # PyTorch data preprocessing
        if isinstance(cfg_train['transforms'], list):
            transformer = build_transformer(cfgs=cfg_train['transforms'],
                                            image_reader=image_reader)
        else:
            transformer = build_common_augmentation(cfg_train['transforms']['type'], cfg_dataset)
        dataset = SOCOProposalDataset(
            root_dir=cfg_train['root_dir'],
            meta_file=cfg_train['meta_file'],
            transform=transformer,
            read_from=cfg_dataset['read_from'],
            image_reader_type=image_reader.get('type', 'pil'),
            soco_type=cfg_train['soco_type'],
            json_dir=cfg_dataset['json_dir'],
        )
    # build sampler
    cfg_train['sampler']['kwargs'] = {}
    cfg_dataset['dataset'] = dataset
    sampler = build_sampler(cfg_train['sampler'], cfg_dataset)
    if cfg_dataset['last_iter'] >= cfg_dataset['max_iter']:
        return {'loader': None}

    # build dataloader
    if cfg_dataset['use_dali']:
        # NVIDIA dali pipeline
        pipeline = ImageNetTrainPipeV2(
            data_root=cfg_train['root_dir'],
            data_list=cfg_train['meta_file'],
            sampler=sampler,
            crop=cfg_dataset['input_size'],
            colorjitter=[0.2, 0.2, 0.2, 0.1]
        )
        loader = DaliDataloader(
            pipeline=pipeline,
            batch_size=cfg_dataset['batch_size'],
            epoch_size=len(sampler),
            num_threads=cfg_dataset['num_workers'],
            last_iter=cfg_dataset['last_iter']
        )
    else:
        # PyTorch dataloader
        loader = DataLoader(
            dataset=dataset,
            batch_size=cfg_dataset['batch_size'],
            shuffle=False,
            num_workers=cfg_dataset['num_workers'],
            pin_memory=True,
            sampler=sampler,
            collate_fn=_collate_fn,
        )
    return {'type': 'train', 'loader': loader}
