import os
import argparse
from easydict import EasyDict
from tensorboardX import SummaryWriter
import pprint
import time, datetime
from prototype.model import Momentum
import torch
import torchvision
import json
import spring.linklink as link
import torch.nn.functional as F
import random
from .base_solver import BaseSolver
from prototype.utils.dist import link_dist, DistModule
from prototype.utils.misc import makedir, create_logger, get_logger, \
    param_group_all, AverageMeter, load_state_model, load_state_optimizer, mixup_data, \
    parse_config
from prototype.model import model_entry
from prototype.optimizer import optim_entry, FP16RMSprop, FP16SGD, FusedFP16SGD, FP16AdamW
from prototype.lr_scheduler import scheduler_entry
from prototype.data import build_soco_train_dataloader
from prototype.utils.grad_clip import clip_grad_norm_, clip_grad_value_, clip_param_grad_value_
import numpy as np
import torch.nn.functional as F


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        # with torch.cuda.stream(self.stream):
        #     for k in self.batch:
        #         if k != 'meta':
        #             self.batch[k] = self.batch[k].to(device=self.opt.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch


class SoCoSolver(BaseSolver):
    def __init__(self, config_file):
        self.config_file = config_file
        self.prototype_info = EasyDict()
        self.config = parse_config(config_file)
        self.setup_env()
        # import ipdb
        # ipdb.set_trace()
        self.build_model()
        self.build_optimizer()
        self.env_load_state()
        self.build_data()
        self.build_lr_scheduler()
        # send_info(self.prototype_info)

    def env_load_state(self):
        # load pretrain checkpoint
        if hasattr(self.config.saver, 'resume') and self.config.saver.resume and len(os.listdir(self.path.save_path))>0:
            last_checkpoint = self.find_last_checkpoint()
            state = torch.load(last_checkpoint, 'cpu')
            self.logger.info(
                f"Recovering from {last_checkpoint}, keys={list(state.keys())}")
            if 'model' in state:
                load_state_model(self.model, state['model'])
            if 'optimizer' in state:
                load_state_optimizer(self.optimizer, state['optimizer'])
            self.state = {}
            self.state['last_iter'] = state['last_iter']
        else:
            self.state = {}
            self.state['last_iter'] = 0

    def setup_env(self):
        # dist
        self.dist = EasyDict()
        self.dist.rank, self.dist.world_size = link.get_rank(), link.get_world_size()
        self.prototype_info.world_size = self.dist.world_size
        # directories
        self.path = EasyDict()
        self.path.root_path = os.path.dirname(self.config_file)
        self.path.save_path = os.path.join(self.path.root_path, 'checkpoints')
        self.path.event_path = os.path.join(self.path.root_path, 'events')
        self.path.result_path = os.path.join(self.path.root_path, 'results')
        makedir(self.path.save_path)
        makedir(self.path.event_path)
        makedir(self.path.result_path)
        # tb_logger
        if self.dist.rank == 0:
            self.tb_logger = SummaryWriter(self.path.event_path)
        # logger
        create_logger(os.path.join(self.path.root_path, 'log.txt'))
        self.logger = get_logger(__name__)
        self.logger.critical(f'config: {pprint.pformat(self.config)}')
        if 'SLURM_NODELIST' in os.environ:
            self.logger.critical(f"hostnames: {os.environ['SLURM_NODELIST']}")
        
        # others
        torch.backends.cudnn.benchmark = True

    def find_last_checkpoint(self):
        ckpt_list = os.listdir(self.path.save_path)
        if 'ckpt.pth.tar' in ckpt_list:
            return 'checkpoints/ckpt.pth.tar'
        elif len(ckpt_list) == 0:
            return None
        num = [int(ckpt.split('.')[0][5:]) for ckpt in ckpt_list]
        num.sort()
        last_checkpoint_path = self.path.save_path+'/ckpt_' + str(num[-1])+'.pth.tar'
        return last_checkpoint_path

    def build_model(self):
        if hasattr(self.config, 'lms'):
            if self.config.lms.enable:
                torch.cuda.set_enabled_lms(True)
                byte_limit = self.config.lms.kwargs.limit * (1 << 30)
                torch.cuda.set_limit_lms(byte_limit)
                self.logger.critical('Enable large model support, limit of {}G!'.format(
                    self.config.lms.kwargs.limit))

        encoder_q = model_entry(self.config.model, use_predictor=True)
        encoder_k = model_entry(self.config.model, use_predictor=False)
        self.model = Momentum(encoder_q, encoder_k, freeze_set=self.config.moco, total_step=self.config.lr_scheduler.kwargs.max_iter)
        self.model.cuda()

        # handle fp16
        if self.config.optimizer.type == 'FP16SGD' or \
           self.config.optimizer.type == 'FusedFP16SGD' or \
           self.config.optimizer.type == 'FP16RMSprop' or \
           self.config.optimizer.type == 'FP16AdamW':
            self.fp16 = True
        else:
            self.fp16 = False

        if self.fp16:
            # if you have modules that must use fp32 parameters, and need fp32 input
            # try use link.fp16.register_float_module(your_module)
            # if you only need fp32 parameters set cast_args=False when call this
            # function, then call link.fp16.init() before call model.half()
            if self.config.optimizer.get('fp16_normal_bn', False):
                self.logger.critical('using normal bn for fp16')
                link.fp16.register_float_module(
                    link.nn.SyncBatchNorm2d, cast_args=False)
                link.fp16.register_float_module(
                    torch.nn.BatchNorm2d, cast_args=False)
            if self.config.optimizer.get('fp16_normal_fc', False):
                self.logger.critical('using normal fc for fp16')
                link.fp16.register_float_module(
                    torch.nn.Linear, cast_args=True)
            link.fp16.init()
            self.model.half()

        self.model = DistModule(self.model, self.config.dist.sync)

    def build_optimizer(self):

        opt_config = self.config.optimizer
        opt_config.kwargs.lr = self.config.lr_scheduler.kwargs.base_lr
        self.prototype_info.optimizer = self.config.optimizer.type

        # make param_groups
        pconfig = {}

        if opt_config.get('no_wd', False):
            pconfig['conv_b'] = {'weight_decay': 0.0}
            pconfig['linear_b'] = {'weight_decay': 0.0}
            pconfig['bn_w'] = {'weight_decay': 0.0}
            pconfig['bn_b'] = {'weight_decay': 0.0}

        if 'pconfig' in opt_config:
            pconfig.update(opt_config['pconfig'])

        if opt_config['type'] == 'AdamW_SGD':
            visual_config = opt_config['visual_config']
            visual_parameters = self.model.module.visual_parameters()
            param_group = []
            if len(visual_parameters) > 0:
                param_group.append(
                    {'params': visual_parameters, **visual_config})
           
            for visual_module in self.model.module.visual_modules():
                param_group_visual, type2num = param_group_all(
                    visual_module, pconfig, visual_config)
                param_group += param_group_visual
        else:
            param_group, type2num = param_group_all(self.model, pconfig)

        opt_config.kwargs.params = param_group

        self.optimizer = optim_entry(opt_config)



    def build_lr_scheduler(self):
        self.prototype_info.lr_scheduler = self.config.lr_scheduler.type
        if not getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.lr_scheduler.kwargs.max_iter = self.config.data.max_iter
        self.config.lr_scheduler.kwargs.optimizer = self.optimizer.optimizer if isinstance(self.optimizer, FP16SGD) or \
            isinstance(self.optimizer, FP16RMSprop) or isinstance(self.optimizer, FP16AdamW) else self.optimizer
        self.config.lr_scheduler.kwargs.last_iter = self.state['last_iter']
        self.lr_scheduler = scheduler_entry(self.config.lr_scheduler)

    def build_data(self):
        # print('='*20, '\n', 'build data')
        test_config = {}
        self.config.data.last_iter = self.state['last_iter']
        test_config['last_iter'] = self.state['last_iter']
        if getattr(self.config.lr_scheduler.kwargs, 'max_iter', False):
            self.config.data.max_iter = self.config.lr_scheduler.kwargs.max_iter
            test_config['max_iter'] = self.config.lr_scheduler.kwargs.max_iter
        else:
            self.config.data.max_epoch = self.config.lr_scheduler.kwargs.max_epoch
            test_config['max_epoch'] = self.config.lr_scheduler.kwargs.max_epoch

        if self.config.data.get('type', 'imagenet') == 'imagenet':
            self.train_data = build_imagenet_train_dataloader(self.config.data)
        elif self.config.data.get('type', 'soco') == 'soco':
            self.train_data = build_soco_train_dataloader(self.config.data)
        elif self.config.data.get('type') == 'clip':
            self.train_data = build_clip_dataloader('train', self.config.data)

        self.prefetch = self.config.data.train.get('prefetch', False)
        if self.prefetch:
            self.prefetcher = DataPrefetcher(self.train_data['loader'])
        elif self.train_data['loader']:
            self.train_data['loader'] = iter(self.train_data['loader'])
        # print('='*20, '\n', 'end build data')

    def pre_train(self):
        self.meters = EasyDict()
        self.meters.batch_time = AverageMeter(self.config.saver.print_freq)
        self.meters.step_time = AverageMeter(self.config.saver.print_freq)
        self.meters.data_time = AverageMeter(self.config.saver.print_freq)
        self.meters.losses = AverageMeter(self.config.saver.print_freq)
        self.meters.loss_reg = AverageMeter(self.config.saver.print_freq)
        self.meters.loss_rpn = AverageMeter(self.config.saver.print_freq)
        self.meters.valloss = AverageMeter(self.config.saver.print_freq)
        
        # self.meters.simsiam_losses = AverageMeter(self.config.saver.print_freq)

        self.model.train()

        # label_smooth = self.config.get('label_smooth', 0.0)
        # self.num_classes = self.config.model.kwargs.get('num_classes', 1000)
        # self.topk = 5 if self.num_classes >= 5 else self.num_classes
        # if label_smooth > 0:
        #     self.logger.info('using label_smooth: {}'.format(label_smooth))
        #     self.criterion = LabelSmoothCELoss(label_smooth, self.num_classes)
        # else:
        #     self.criterion = torch.nn.CrossEntropyLoss()
                       
        # self.mixup = self.config.get('mixup', 1.0)
        # self.cutmix = self.config.get('cutmix', 0.0)
        # if self.mixup < 1.0:
        #     self.logger.critical(
        #         'using mixup with alpha of: {}'.format(self.mixup))
        # if self.cutmix > 0.0:
        #     self.logger.critical(
        #         'using cutmix with alpha of: {}'.format(self.cutmix))

    def train(self):
        self.pre_train()
        total_step = len(self.train_data['loader'])
        self.model.total_step = total_step
        start_step = self.state['last_iter'] + 1
        end = time.time()

        for i in range(len(self.train_data['loader'])):
            if self.prefetch:
                batch = self.prefetcher.next()
            else:
                batch = next(self.train_data['loader'])

            curr_step = start_step + i
            self.lr_scheduler.step(curr_step)
            # lr_scheduler.get_lr()[0] is the main lr
            current_lr = self.lr_scheduler.get_lr()[0]
            # measure data loading time
            self.meters.data_time.update(time.time() - end)
            # set_trace()
            if self.fp16:
                batch['view1'] = batch['view1'].cuda().half()
                batch['view2'] = batch['view2'].cuda().half()
                batch['view3'] = batch['view3'].cuda().half()
                batch['proposal_bbox1'] = [b.cuda().half() for b in batch['proposal_bbox1']]
                batch['proposal_bbox2'] = [b.cuda().half() for b in batch['proposal_bbox2']]
                batch['proposal_bbox3'] = [b.cuda().half() for b in batch['proposal_bbox3']]
            else:
                batch['view1'] = batch['view1'].cuda()
                batch['view2'] = batch['view2'].cuda()
                batch['view3'] = batch['view3'].cuda()
                batch['proposal_bbox1'] = [b.cuda() for b in batch['proposal_bbox1']]
                batch['proposal_bbox2'] = [b.cuda() for b in batch['proposal_bbox2']]
                batch['proposal_bbox3'] = [b.cuda() for b in batch['proposal_bbox3']]
            
            output = self.model(batch,step=curr_step)
            loss = -2*torch.cosine_similarity(output['online_q1'], output['target_k1'], dim=1).mean() / self.dist.world_size \
                    + -2*torch.cosine_similarity(output['online_q1'], output['target_k2'], dim=1).mean() / self.dist.world_size \
                    + -2*torch.cosine_similarity(output['target_q1'], output['online_k1'], dim=1).mean() / self.dist.world_size \
                    + -2*torch.cosine_similarity(output['target_q1'], output['online_k2'], dim=1).mean() / self.dist.world_size 
            self.meters.losses.reduce_update(loss.clone())
            if self.config.model.get('loss_reg_scale', False):
                loss_reg_scale = self.config.model.loss_reg_scale
            else:
                loss_reg_scale = 20
            if 'proposal_bbox1_gt' in batch.keys():
                self.meters.loss_reg.reduce_update(output['loss_reg'].clone())
                loss += loss_reg_scale * output['loss_reg'] 
            if 'rpn_loss_cls' in output.keys():
                loss_rpn = output['rpn_loss_cls'] + output['rpn_loss_bbox']
                self.meters.loss_rpn.reduce_update(loss_rpn.clone())
                loss += loss_reg_scale * loss_rpn
                
            
            # compute and update gradient
            self.optimizer.zero_grad()
            if FusedFP16SGD is not None and isinstance(self.optimizer, FusedFP16SGD):
                self.optimizer.backward(loss)
                if self.config.grad_clip.type == 'norm':
                    clip_grad_norm_(self.model.parameters(),
                                    self.config.grad_clip.value)
                    # set_trace()
                elif self.config.grad_clip.type == 'value':
                    clip_grad_value_(self.model.parameters(),
                                     self.config.grad_clip.value)
                elif self.config.grad_clip.type == 'logit_scale_grad':
                    clip_param_grad_value_(
                        self.model.module.logit_scale, self.config.grad_clip.value)
                elif self.config.grad_clip.type == 'logit_scale_param':
                    before = self.model.module.logit_scale.data.item()
                elif self.config.grad_clip.type == 'constant':
                    self.model.module.logit_scale.requires_grad = False
                self.model.sync_gradients()
                self.optimizer.step()
                if self.config.grad_clip.type == 'logit_scale_param':
                    after = self.model.module.logit_scale.data.item()
                    tem = self.model.module.logit_scale.data
                    if (after-before) > self.config.grad_clip.value:
                        self.model.module.logit_scale.data = torch.as_tensor(
                            before+self.config.grad_clip.value, dtype=tem.dtype, device=tem.device)
                    elif (before-after) > self.config.grad_clip.value:
                        self.model.module.logit_scale.data = torch.as_tensor(
                            before-self.config.grad_clip.value, dtype=tem.dtype, device=tem.device)
            elif isinstance(self.optimizer, FP16SGD) or isinstance(self.optimizer, FP16RMSprop):

                def closure():
                    self.optimizer.backward(loss, False)
                    self.model.sync_gradients()
                    # check overflow, convert to fp32 grads, downscale
                    self.optimizer.update_master_grads()
                    return loss
                self.optimizer.step(closure)
            else:
                loss.backward()  #retain_graph=False
                # set_trace()
                self.model.sync_gradients()
                if self.config.grad_clip.type == 'norm':
                    clip_grad_norm_(self.model.parameters(),
                                    self.config.grad_clip.value)
                elif self.config.grad_clip.type == 'value':
                    clip_grad_value_(self.model.parameters(),
                                     self.config.grad_clip.value)
                elif self.config.grad_clip.type == 'logit_scale_grad':
                    clip_param_grad_value_(
                        self.model.module.logit_scale, self.config.grad_clip.value)
                elif self.config.grad_clip.type == 'logit_scale_param':
                    before = self.model.module.logit_scale.data.item()
                elif self.config.grad_clip.type == 'constant':
                    self.model.module.logit_scale.requires_grad = False
                elif self.config.grad_clip.type == 'logit_scale_param_abs_min':
                    self.model.module.logit_scale.data.clamp_(min=self.config.grad_clip.value)
                self.optimizer.step()
                # set_trace()
                if self.config.grad_clip.type == 'logit_scale_param':
                    after = self.model.module.logit_scale.data.item()
                    tem = self.model.module.logit_scale.data
                    if (after-before) > self.config.grad_clip.value:
                        self.model.module.logit_scale.data = torch.as_tensor(
                            before+self.config.grad_clip.value, dtype=tem.dtype, device=tem.device)
                    elif (before-after) > self.config.grad_clip.value:
                        self.model.module.logit_scale.data = torch.as_tensor(
                            before-self.config.grad_clip.value, dtype=tem.dtype, device=tem.device)

            # clamp
            if self.config.grad_clip.type == 'logit_scale_param_ema':
                # if self.dist.rank == 0:
                #     print('*****************before_buffer', logit_scale.buffer)
                #     print('*****************before_param', logit_scale.param)
                logit_scale.clamp()
                logit_scale.update()

            # measure elapsed time
            self.meters.batch_time.update(time.time() - end)
            # training logger
            if curr_step % self.config.saver.print_freq == 0 and self.dist.rank == 0:
                self.tb_logger.add_scalar(
                    'loss_train', self.meters.losses.avg, curr_step)
                self.tb_logger.add_scalar(
                    'loss_reg', self.meters.loss_reg.avg, curr_step)
                self.tb_logger.add_scalar(
                    'loss_rpn', self.meters.loss_rpn.avg, curr_step)
                self.tb_logger.add_scalar('lr', current_lr, curr_step)
                # self.tb_logger.add_scalar('loss_reg_scale', loss_reg_scale, curr_step)
                remain_secs = (total_step - curr_step) * \
                    self.meters.batch_time.avg
                remain_time = datetime.timedelta(seconds=round(remain_secs))
                finish_time = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(time.time()+remain_secs))
                log_msg = f'Iter: [{curr_step}/{total_step}]\t' \
                    f'Time {self.meters.batch_time.val:.3f} ({self.meters.batch_time.avg:.3f})\t' \
                    f'Data {self.meters.data_time.val:.3f} ({self.meters.data_time.avg:.3f})\t' \
                    f'Loss {self.meters.losses.val:.4f} ({self.meters.losses.avg:.4f})\t' \
                    f'Loss_reg {self.meters.loss_reg.val:.4f} ({self.meters.loss_reg.avg:.4f})\t' \
                    f'loss_reg_scale {loss_reg_scale:.4f} ({loss_reg_scale:.4f})\t' \
                    f'Loss_rpn {self.meters.loss_rpn.val:.4f} ({self.meters.loss_rpn.avg:.4f})\t' \
                    f'LR {current_lr:.4f}\t'

                self.logger.critical(log_msg)   
                   
            if curr_step > 0 and curr_step % self.config.saver.save_freq == 0:
                # save ckpt
                if self.dist.rank == 0:
                    if self.config.saver.save_many:
                        ckpt_name = f'{self.path.save_path}/ckpt_{curr_step}.pth.tar'
                    else:
                        ckpt_name = f'{self.path.save_path}/ckpt.pth.tar'
                    state = {}
                    state['model'] = self.model.state_dict()
                    state['optimizer'] = self.optimizer.state_dict()
                    state['last_iter'] = curr_step
                    torch.save(state, ckpt_name)

            end = time.time()



@link_dist
def main():
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()
    # build solver
    import prototype.solver.crash_on_ipy
    solver = SoCoSolver(args.config)
    # evaluate or train
    if args.evaluate:
        if not hasattr(solver.config.saver, 'pretrain'):
            solver.logger.warn(
                'Evaluating without resuming any solver checkpoints.')
        for id, val_data in enumerate(solver.val_data):
            solver.evaluate(val_data)
            if solver.ema is not None:
                solver.ema.load_ema(solver.model)
                solver.evaluate(val_data)
    else:
        if solver.config.data.last_iter < solver.config.data.max_iter:
            solver.train()
        else:
            solver.logger.info('Training has been completed to max_iter!')


if __name__ == '__main__':
    main()
