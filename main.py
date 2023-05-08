#!/usr/bin/env python

import argparse
import ast
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tqdm import tqdm

import spatial_transforms
from NCEAverage_m import NCEAverage
from NCECriterion import NCECriterion
from dataloader import create_train_loader, create_val_loader
from dataset import DAD
from dataset_test import DAD_Test
from model import generate_model
from models.MoCo_MTV_2 import MTV

from test_MTV import get_normal_vector, split_acc_diff_threshold, cal_score
from utils import adjust_learning_rate, AverageMeter, Logger, get_fusion_label, l2_normalize, post_process, evaluate, \
    get_score, print_args
from utils import select_device


def parse_args():
    parser = argparse.ArgumentParser(description='DAD training on Videos')
    parser.add_argument('--root_path', default='/data1/sgj/Datasets/DAD/DAD/', type=str,
                        help='root path of the dataset')
    parser.add_argument('--mode', default='train', type=str, help='train | test(validation)')
    parser.add_argument('--view', default='front_depth', type=str, help='front_depth | front_IR | top_depth | top_IR')
    parser.add_argument('--feature_dim', default=128, type=int, help='To which dimension will video clip be embedded')
    parser.add_argument('--sample_duration', default=16, type=int, help='Temporal duration of each video clip')
    parser.add_argument('--sample_size', default=112, type=int, help='Height and width of inputs')
    parser.add_argument('--model_type', default='resnet', type=str, help='so far only resnet')
    parser.add_argument('--model_depth', default=18, type=int, help='Depth of resnet (18 | 50 | 101)')
    parser.add_argument('--shortcut_type', default='A', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--pre_train_model', default=True, type=ast.literal_eval, help='Whether use pre-trained model')
    parser.add_argument('--use_cuda', default=True, type=ast.literal_eval, help='If true, cuda is used.')
    parser.add_argument('--n_train_batch_size', default=10, type=int, help='Batch Size for normal training data')
    parser.add_argument('--a_train_batch_size', default=140, type=int, help='Batch Size for anormal training data')
    parser.add_argument('--batch_size', default=150, type=int, help='Batch Size for anormal training data')
    parser.add_argument('--val_batch_size', default=70, type=int, help='Batch Size for validation data')
    parser.add_argument('--learning_rate', default=0.01, type=float,
                        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening', default=0.0, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight Decay')
    parser.add_argument('--epochs', default=110, type=int, help='Number of total epochs to run')
    parser.add_argument('--n_threads', default=4, type=int, help='num of workers loading dataset')
    parser.add_argument('--tracking', default=True, type=ast.literal_eval,
                        help='If true, BN uses tracking running stats')
    parser.add_argument('--norm_value', default=255, type=int,
                        help='If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument('--cal_vec_batch_size', default=100, type=int,
                        help='batch size for calculating normal driving average vector.')
    parser.add_argument('--tau', default=0.1, type=float,
                        help='a temperature parameter that controls the concentration level of the distribution of embedded vectors')
    parser.add_argument('--manual_seed', default=3407, type=int, help='Manually set random seed')
    parser.add_argument('--memory_bank_size', default=200, type=int, help='Memory bank size')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--lr_decay', default=100, type=int,
                        help='Number of epochs after which learning rate will be reduced to 1/10 of original value')
    # parser.add_argument('--resume_path', default='resnet_front_depth_150_pre.pth', type=str, help='path of previously trained model')
    # parser.add_argument('--resume_head_path', default='resnet_front_depth_150_head.pth', type=str, help='path of previously trained model head')
    parser.add_argument('--resume_path', default='', type=str, help='path of previously trained model')
    parser.add_argument('--resume_head_path', default='', type=str, help='path of previously trained model head')
    parser.add_argument('--resume_line_path', default='', type=str, help='path of previously trained model head')
    parser.add_argument('--initial_scales', default=1.0, type=float, help='Initial scale for multiscale cropping')
    parser.add_argument('--scale_step', default=0.9, type=float, help='Scale step for multiscale cropping')
    parser.add_argument('--n_scales', default=5, type=int, help='Number of scales for multiscale cropping')
    parser.add_argument('--train_crop', default='random', type=str,
                        help='Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)')
    parser.add_argument('--checkpoint_folder', default='exp/MTV-fd-i2t+t2i-cl-fused_model-1tr_layer-cat_3d_feat-DAD_NCE/', type=str,
                        help='folder to store checkpoints')
    parser.add_argument('--log_folder', default='./logs/', type=str, help='folder to store log files')
    parser.add_argument('--log_resume', default=False, type=ast.literal_eval,
                        help='True|False: a flag controlling whether to create a new log file')
    parser.add_argument('--normvec_folder', default='./normvec/', type=str, help='folder to store norm vectors')
    parser.add_argument('--score_folder', default='./score/', type=str, help='folder to store scores')
    parser.add_argument('--Z_momentum', default=0.9, help='momentum for normalization constant Z updates')
    parser.add_argument('--groups', default=3, type=int, help='hyper-parameters when using shufflenet')
    parser.add_argument('--width_mult', default=2.0, type=float,
                        help='hyper-parameters when using shufflenet|mobilenet')
    parser.add_argument('--val_step', default=1, type=int, help='validate per val_step epochs')
    parser.add_argument('--downsample', default=2, type=int, help='Downsampling. Select 1 frame out of N')
    parser.add_argument('--save_step', default=10, type=int, help='checkpoint will be saved every save_step epochs')
    parser.add_argument('--n_split_ratio', default=1.0, type=float,
                        help='the ratio of normal driving samples will be used during training')
    parser.add_argument('--a_split_ratio', default=1.0, type=float,
                        help='the ratio of normal driving samples will be used during training')
    parser.add_argument('--window_size', default=6, type=int, help='the window size for post-processing')
    parser.add_argument('--device', default='3', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--world_size', default=1, help='the number of devices ')
    parser.add_argument('--sync_bn', action='store_true', default=True,
                        help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--mask', action='store_true', default=False,
                        help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--cbam', action='store_true', default=False,
                        help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--use_ssmctb', action='store_true', default=False,
                        help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--host', default='23455', type=str,
                        help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--kv', default='frame', type=str,
                        help='which model feature becomes the kv of decoder')
    parser.add_argument('--num_tr_layer', default=1, type=int,
                        help='which model feature becomes the kv of decoder')  # num_tr_layer
    args = parser.parse_args()
    return args


def train(train_normal_loader, train_anormal_loader, model_mtv, nce_average,
          criterion, optimizer, epoch, args, batch_logger, epoch_logger, rank, scheduler, memory_bank=None):
    losses = AverageMeter()
    prob_meter = AverageMeter()
    model_mtv.train()
    train_n_a_loader = zip(train_normal_loader, train_anormal_loader)
    train_bar = tqdm(train_n_a_loader)
    for batch, ((normal_data_clip, idx_n), (anormal_data_clip, idx_a)) in enumerate(train_bar):

        data_clip = torch.cat((normal_data_clip, anormal_data_clip), dim=0)

        if args.use_cuda:
            data_clip = data_clip.to(rank)
            normal_data_clip = normal_data_clip.to(rank)

        # ================forward====================
        normed_vec, itc_loss = model_mtv(data_clip)  # batchsize*512
        # normed_vec = model_mtv(data_clip)  # batchsize*512

        n_vec = normed_vec[0:args.n_train_batch_size]  #
        a_vec = normed_vec[args.n_train_batch_size:]
        outs, probs = nce_average(n_vec, a_vec)
        # loss = memory_queue(n_vec,a_vec)
        loss = criterion(outs)
        loss += itc_loss
        # ================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
        # ===========update memory bank===============
        model_mtv.eval()
        # n = model_mtv(normal_data_clip)
        n, _ = model_mtv(normal_data_clip)
        n = n.detach()
        average = torch.mean(n, dim=0, keepdim=True)
        if len(memory_bank) < args.memory_bank_size:
            memory_bank.append(average)
        else:
            memory_bank.pop(0)
            memory_bank.append(average)
        model_mtv.train()

        # ===============update meters ===============
        losses.update(loss.item(), 4830)
        # prob_meter.update(probs.item(), outs.size(0))

        # =================logging=====================
        batch_logger.log({
            'epoch': epoch,
            'batch': batch,
            'loss': losses.val,
            'probs': prob_meter.val,
            'lr': optimizer.param_groups[0]['lr']
        })
        train_bar.desc = f'Training Process is running: {epoch}/{args.epochs}  | Batch: {batch} | Loss: {losses.val:.6f} ({losses.avg:.6f}))'
        # print(
        #     f'Training Process is running: {epoch}/{args.epochs}  | Batch: {batch} | Loss: {losses.val} ({losses.avg}) | Probs: {prob_meter.val} ({prob_meter.avg})')

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'probs': prob_meter.avg,
        'lr': optimizer.param_groups[0]['lr']
    })
    ##torch.cuda.empty_cache()
    return memory_bank, losses.avg


def train_from_scratch(model, rank, world_size, dampening, len_neg,
                       len_pos, args):
    # len_neg = 2240
    if args.sync_bn and args.use_cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)
        print('Using SyncBatchNorm()')

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank, static_graph=True
    )

    optimizer = torch.optim.SGD(list(model.parameters()),
                                lr=args.learning_rate, momentum=args.momentum,
                                dampening=dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)

    # scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)
    scheduler = None
    nce_average = NCEAverage(args.feature_dim, len_neg, len_pos, args.tau, args.Z_momentum)
    criterion = NCECriterion(len_neg)
    begin_epoch = 1
    best_acc = 0
    memory_bank = []
    return model, optimizer, nce_average, criterion, begin_epoch, best_acc, scheduler, memory_bank


def train_from_resume(model, model_head, rank, world_size, dampening, len_neg, len_pos, args):
    if args.sync_bn and args.use_cuda and rank != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)
        print('Using SyncBatchNorm()')

    args.pre_train_model = False
    resume_path = os.path.join(args.checkpoint_folder, args.resume_path)
    resume_checkpoint = torch.load(resume_path)
    resume_head_checkpoint = torch.load(os.path.join(args.checkpoint_folder, args.resume_head_path))
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[rank], output_device=rank,
    )
    model_head = torch.nn.parallel.DistributedDataParallel(
        model_head, device_ids=[rank], output_device=rank,
    )

    model.load_state_dict(resume_checkpoint['state_dict'])
    model_head.load_state_dict(resume_head_checkpoint['state_dict'])

    if args.use_cuda:
        model.cuda()
        model_head.cuda()
    optimizer = torch.optim.SGD(list(model.parameters()) + list(model_head.parameters()), lr=args.learning_rate,
                                momentum=args.momentum,
                                dampening=dampening, weight_decay=args.weight_decay, nesterov=args.nesterov)

    if "optimizer" in resume_checkpoint.keys():
        optimizer.load_state_dict(resume_checkpoint['optimizer'])

    nce_average = resume_checkpoint['nce_average'].to(rank)
    begin_epoch = resume_checkpoint['epoch'] + 1
    best_acc = resume_checkpoint['acc']
    memory_bank = resume_checkpoint['memory_bank']
    memory_bank = [memory.to(rank) for memory in memory_bank]
    criterion = NCECriterion(len_neg)
    del resume_checkpoint
    torch.cuda.empty_cache()
    adjust_learning_rate(optimizer, args.learning_rate)

    return model, model_head, optimizer, nce_average, begin_epoch, best_acc, memory_bank, criterion


def main_train(rank, world_size, args):
    if args.nesterov:
        dampening = 0
    else:
        dampening = args.dampening

    device = select_device(args.device, batch_size=args.batch_size)
    if rank != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        assert args.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert args.batch_size % world_size == 0, f'--batch-size {args.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > rank, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", rank=rank,
                                world_size=world_size, init_method='tcp://127.0.0.1:' + args.host)

    print(
        "====================== ===========Loading Anormal-Driving Training Data!=================================")

    train_anormal_loader, training_anormal_data = create_train_loader(args, type="anormal")
    print("=================================Loading Normal-Driving Training Data!=================================")

    train_normal_loader, training_normal_data = create_train_loader(args, type="normal")
    print("========================================Loading Validation Data========================================")

    validation_loader, validation_data = create_val_loader(args)
    len_neg = training_anormal_data.__len__()
    len_pos = training_normal_data.__len__()

    print(f'len_neg: {len_neg}')
    print(f'len_pos: {len_pos}')
    print(
        "============================================Generating Model============================================")

    model_mtv = MTV(args)
    model_mtv.cuda()

    if args.resume_path == '':
        # =============== train without previously trained model ===============
        model_mtv, optimizer, nce_average, criterion, begin_epoch, best_acc, scheduler, memory_bank = \
            train_from_scratch(model_mtv, rank, world_size, dampening, len_neg, len_pos, args)

    else:
        # =============== load previously trained model ===============

        model_mtv, optimizer, nce_average, begin_epoch, best_acc, memory_bank, criterion = \
            train_from_resume(model_mtv, rank, world_size, dampening, len_neg, len_pos, args)

    print(
        "==========================================!!!START TRAINING!!!==========================================")

    batch_logger = Logger(os.path.join(args.log_folder, f'batch_{args.checkpoint_folder.split("/")[1]}.log'),
                          ['epoch', 'batch', 'loss', 'probs', 'lr'],
                          args.log_resume)
    epoch_logger = Logger(os.path.join(args.log_folder, f'epoch_{args.checkpoint_folder.split("/")[1]}.log'),
                          ['epoch', 'loss', 'probs', 'lr'],
                          args.log_resume)
    val_logger = Logger(os.path.join(args.log_folder, f'val_{args.checkpoint_folder.split("/")[1]}.log'),
                        ['epoch', 'accuracy', 'auc','normal_acc', 'anormal_acc', 'threshold'], args.log_resume)
    best_logger = Logger(os.path.join(args.log_folder, f'best_{args.checkpoint_folder.split("/")[1]}.log'),
                        ['epoch', 'accuracy', 'auc','normal_acc', 'anormal_acc', 'threshold'], args.log_resume)
    for epoch in range(begin_epoch, begin_epoch + args.epochs + 1):
        train_normal_loader.sampler.set_epoch(epoch)
        train_anormal_loader.sampler.set_epoch(epoch)
        memory_bank, loss = train(train_normal_loader, train_anormal_loader, model_mtv, nce_average, criterion,
                                  optimizer, epoch, args, batch_logger,
                                  epoch_logger, rank, scheduler, memory_bank)

        if epoch % args.val_step == 0:

            print(
                "==========================================!!!Evaluating!!!==========================================")
            normal_vec = torch.mean(torch.cat(memory_bank, dim=0), dim=0, keepdim=True)
            normal_vec = l2_normalize(normal_vec).cuda()

            model_mtv.eval()
            accuracy, best_threshold, acc_n, acc_a, acc_list, acc_n_list, acc_a_list,auc = split_acc_diff_threshold(
                model_mtv, normal_vec, validation_loader, args.use_cuda,rank, args)
            print(
                f'Epoch: {epoch}/{args.epochs} | Accuracy: {accuracy} |AUC:{auc}| Normal Acc: {acc_n} | Anormal Acc: {acc_a} | Threshold: {best_threshold}')
            # with torch_distributed_zero_first(rank):
            print(
                "==========================================!!!Logging!!!==========================================")

            val_logger.log({
                'epoch': epoch,
                'accuracy': accuracy * 100,
                'auc':auc,
                'normal_acc': acc_n * 100,
                'anormal_acc': acc_a * 100,
                'threshold': best_threshold
            })
            if accuracy > best_acc:
                best_acc = accuracy
                print(
                    "==========================================!!!Saving!!!==========================================")
                checkpoint_path = os.path.join(args.checkpoint_folder,
                                               f'best_model_mtv_{args.model_type}_{args.view}.pth')
                states = {
                    'epoch': epoch,
                    'state_dict': model_mtv.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'acc': accuracy,
                    'auc': auc,
                    'threshold': best_threshold,
                    'nce_average': nce_average,
                    'memory_bank': memory_bank
                }
                torch.save(states, checkpoint_path)

                print(
                    "==========================================!!!Logging!!!==========================================")

                best_logger.log({
                    'epoch': epoch,
                    'accuracy': accuracy * 100,
                    'auc': auc,
                    'normal_acc': acc_n * 100,
                    'anormal_acc': acc_a * 100,
                    'threshold': best_threshold
                })
            if epoch % args.save_step == 0:
                print(
                    "==========================================!!!Saving!!!==========================================")
                checkpoint_path = os.path.join(args.checkpoint_folder,
                                               f'last_model_mtv_{args.model_type}_{args.view}.pth')
                states = {
                    'epoch': epoch,
                    'state_dict': model_mtv.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'acc': accuracy,
                    'auc': auc,
                    'threshold': best_threshold,
                    'nce_average': nce_average,
                    'memory_bank': memory_bank
                }
                torch.save(states, checkpoint_path)
                ########################################################################################

        if epoch % args.lr_decay == 0:
            lr = args.learning_rate * (0.1 ** (epoch // args.lr_decay))
            adjust_learning_rate(optimizer, lr)
            print(f'New learning rate: {lr}')


if __name__ == '__main__':
    args = parse_args()

    print_args(vars(args))
    if not os.path.exists(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)
    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)
    if not os.path.exists(args.normvec_folder):
        os.makedirs(args.normvec_folder)
    if not os.path.exists(args.score_folder):
        os.makedirs(args.score_folder)
    torch.manual_seed(args.manual_seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.manual_seed)

    if args.mode == 'train':
        world_size = args.world_size
        mp.spawn(
            main_train,
            args=(world_size, args,),
            nprocs=world_size,
            join=True
        )


    elif args.mode == 'test':
        if args.nesterov:
            dampening = 0
        else:
            dampening = args.dampening
        if not os.path.exists(args.normvec_folder):
            os.makedirs(args.normvec_folder)
        score_folder = './score/'
        if not os.path.exists(score_folder):
            os.makedirs(score_folder)
        args.pre_train_model = False
        '''
        #### if you want to test your modified code please download DAD dataset, placing the weights to the right path
        and use this part to calculate the sim score###
        #### 如果你想要test你的模型，请下载DAD数据集，并且将自己的模型权重放到正确的路径中并且用这部分代码去计算sim score###
        model_front_d = generate_model(args)
        model_front_ir = generate_model(args)
        model_top_d = generate_model(args)
        model_top_ir = generate_model(args)

        resume_path_front_d = './best_checkpoints/best_model_mtv_resnet_front_depth.pth'
        resume_path_front_ir = './best_checkpoints/best_model_mtv_resnet_front_IR.pth'
        resume_path_top_d = './best_checkpoints/best_model_mtv_resnet_top_depth.pth'
        resume_path_top_ir = './best_checkpoints/best_model_mtv_resnet_top_IR.pth'

        resume_checkpoint_front_d = torch.load(resume_path_front_d)
        resume_checkpoint_front_ir = torch.load(resume_path_front_ir)
        resume_checkpoint_top_d = torch.load(resume_path_top_d)
        resume_checkpoint_top_ir = torch.load(resume_path_top_ir)

        model_front_d.load_state_dict(
            {k.replace('module.', ''): v for k, v in resume_checkpoint_front_d['state_dict'].items()})
        model_front_ir.load_state_dict(
            {k.replace('module.', ''): v for k, v in resume_checkpoint_front_ir['state_dict'].items()})
        model_top_d.load_state_dict(
            {k.replace('module.', ''): v for k, v in resume_checkpoint_top_d['state_dict'].items()})
        model_top_ir.load_state_dict(
            {k.replace('module.', ''): v for k, v in resume_checkpoint_top_ir['state_dict'].items()})

        # model_front_d.load_state_dict(resume_checkpoint_front_d['state_dict'])
        # model_front_ir.load_state_dict(resume_checkpoint_front_ir['state_dict'])
        # model_top_d.load_state_dict(resume_checkpoint_top_d['state_dict'])
        # model_top_ir.load_state_dict(resume_checkpoint_top_ir['state_dict'])

        model_front_d.eval()
        model_front_ir.eval()
        model_top_d.eval()
        model_top_ir.eval()

        val_spatial_transform = spatial_transforms.Compose([
            spatial_transforms.Scale(args.sample_size),
            spatial_transforms.CenterCrop(args.sample_size),
            spatial_transforms.ToTensor(args.norm_value),
            spatial_transforms.Normalize([0], [1]),
        ])

        print("========================================Loading Test Data========================================")
        test_data_front_d = DAD_Test(root_path=args.root_path,
                                     subset='validation',
                                     view='front_depth',
                                     sample_duration=args.sample_duration,
                                     type=None,
                                     spatial_transform=val_spatial_transform,
                                     )
        test_loader_front_d = torch.utils.data.DataLoader(
            test_data_front_d,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        num_val_data_front_d = test_data_front_d.__len__()
        print('Front depth view is done')

        test_data_front_ir = DAD_Test(root_path=args.root_path,
                                      subset='validation',
                                      view='front_IR',
                                      sample_duration=args.sample_duration,
                                      type=None,
                                      spatial_transform=val_spatial_transform,
                                      )
        test_loader_front_ir = torch.utils.data.DataLoader(
            test_data_front_ir,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        num_val_data_front_ir = test_data_front_ir.__len__()
        print('Front IR view is done')

        test_data_top_d = DAD_Test(root_path=args.root_path,
                                   subset='validation',
                                   view='top_depth',
                                   sample_duration=args.sample_duration,
                                   type=None,
                                   spatial_transform=val_spatial_transform,
                                   )
        test_loader_top_d = torch.utils.data.DataLoader(
            test_data_top_d,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        num_val_data_top_d = test_data_top_d.__len__()
        print('Top depth view is done')

        test_data_top_ir = DAD_Test(root_path=args.root_path,
                                    subset='validation',
                                    view='top_IR',
                                    sample_duration=args.sample_duration,
                                    type=None,
                                    spatial_transform=val_spatial_transform,
                                    )
        test_loader_top_ir = torch.utils.data.DataLoader(
            test_data_top_ir,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        num_val_data_top_ir = test_data_top_ir.__len__()
        print('Top IR view is done')
        assert num_val_data_front_d == num_val_data_front_ir == num_val_data_top_d == num_val_data_top_ir

        print("==========================================Loading Normal Data==========================================")
        training_normal_data_front_d = DAD(root_path=args.root_path,
                                           subset='train',
                                           view='front_depth',
                                           sample_duration=args.sample_duration,
                                           type='normal',
                                           spatial_transform=val_spatial_transform,
                                           )

        training_normal_size = int(len(training_normal_data_front_d) * args.n_split_ratio)
        training_normal_data_front_d = torch.utils.data.Subset(training_normal_data_front_d,
                                                               np.arange(training_normal_size))

        train_normal_loader_for_test_front_d = torch.utils.data.DataLoader(
            training_normal_data_front_d,
            batch_size=args.cal_vec_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        print(f'Front depth view is done (size: {len(training_normal_data_front_d)})')

        training_normal_data_front_ir = DAD(root_path=args.root_path,
                                            subset='train',
                                            view='front_IR',
                                            sample_duration=args.sample_duration,
                                            type='normal',
                                            spatial_transform=val_spatial_transform,
                                            )

        training_normal_size = int(len(training_normal_data_front_ir) * args.n_split_ratio)
        training_normal_data_front_ir = torch.utils.data.Subset(training_normal_data_front_ir,
                                                                np.arange(training_normal_size))

        train_normal_loader_for_test_front_ir = torch.utils.data.DataLoader(
            training_normal_data_front_ir,
            batch_size=args.cal_vec_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        print(f'Front IR view is done (size: {len(training_normal_data_front_ir)})')

        training_normal_data_top_d = DAD(root_path=args.root_path,
                                         subset='train',
                                         view='top_depth',
                                         sample_duration=args.sample_duration,
                                         type='normal',
                                         spatial_transform=val_spatial_transform,
                                         )

        training_normal_size = int(len(training_normal_data_top_d) * args.n_split_ratio)
        training_normal_data_top_d = torch.utils.data.Subset(training_normal_data_top_d,
                                                             np.arange(training_normal_size))

        train_normal_loader_for_test_top_d = torch.utils.data.DataLoader(
            training_normal_data_top_d,
            batch_size=args.cal_vec_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        print(f'Top depth view is done (size: {len(training_normal_data_top_d)})')

        training_normal_data_top_ir = DAD(root_path=args.root_path,
                                          subset='train',
                                          view='top_IR',
                                          sample_duration=args.sample_duration,
                                          type='normal',
                                          spatial_transform=val_spatial_transform,
                                          )

        training_normal_size = int(len(training_normal_data_top_ir) * args.n_split_ratio)
        training_normal_data_top_ir = torch.utils.data.Subset(training_normal_data_top_ir,
                                                              np.arange(training_normal_size))

        train_normal_loader_for_test_top_ir = torch.utils.data.DataLoader(
            training_normal_data_top_ir,
            batch_size=args.cal_vec_batch_size,
            shuffle=True,
            num_workers=args.n_threads,
            pin_memory=True,
        )
        print(f'Top IR view is done (size: {len(training_normal_data_top_ir)})')

        print(
            "============================================START EVALUATING============================================")
        normal_vec_front_d = get_normal_vector(model_front_d, train_normal_loader_for_test_front_d,
                                               args.cal_vec_batch_size,
                                               args.feature_dim,
                                               args.use_cuda)
        np.save(os.path.join(args.normvec_folder, 'normal_vec_front_d.npy'), normal_vec_front_d.cpu().numpy())

        normal_vec_front_ir = get_normal_vector(model_front_ir, train_normal_loader_for_test_front_ir,
                                                args.cal_vec_batch_size,
                                                args.feature_dim,
                                                args.use_cuda)
        np.save(os.path.join(args.normvec_folder, 'normal_vec_front_ir.npy'), normal_vec_front_ir.cpu().numpy())

        normal_vec_top_d = get_normal_vector(model_top_d, train_normal_loader_for_test_top_d, args.cal_vec_batch_size,
                                             args.feature_dim,
                                             args.use_cuda)
        np.save(os.path.join(args.normvec_folder, 'normal_vec_top_d.npy'), normal_vec_top_d.cpu().numpy())

        normal_vec_top_ir = get_normal_vector(model_top_ir, train_normal_loader_for_test_top_ir,
                                              args.cal_vec_batch_size,
                                              args.feature_dim,
                                              args.use_cuda)
        np.save(os.path.join(args.normvec_folder, 'normal_vec_top_ir.npy'), normal_vec_top_ir.cpu().numpy())

        
        cal_score(model_front_d, model_front_ir, model_top_d, model_top_ir,
                  normal_vec_front_d, normal_vec_front_ir, normal_vec_top_d, normal_vec_top_ir,
                  test_loader_front_d, test_loader_front_ir,test_loader_top_d,
                  test_loader_top_ir, score_folder, args.use_cuda)
        '''

        #gt = get_fusion_label(os.path.join(args.root_path, 'LABEL.csv'))
        gt = get_fusion_label('LABEL.csv'))
        hashmap = {'top_d': 'Top(D)',
                   'top_ir': 'Top(IR)',
                   'fusion_top': 'Top(DIR)',
                   'front_d': 'Front(D)',
                   'front_ir': 'Front(IR)',
                   'fusion_front': 'Front(DIR)',
                   'fusion_d': 'Fusion(D)',
                   'fusion_ir': 'Fusion(IR)',
                   'fusion_all': 'Fusion(DIR)'
                   }

        for mode, mode_name in hashmap.items():
            score = get_score(score_folder, mode)
            best_acc, best_threshold, AUC = evaluate(score, gt, False)
            print(
                f'Mode: {mode_name}:      Best Acc: {round(best_acc, 2)} | Threshold: {round(best_threshold, 2)} | AUC: {round(AUC, 4)}')
            score = post_process(score, args.window_size)
            best_acc, best_threshold, AUC = evaluate(score, gt, False)
            print(
                f'View: {mode_name}(post-processed):       Best Acc: {round(best_acc, 2)} | Threshold: {round(best_threshold, 2)} | AUC: {round(AUC, 4)} \n')

    # main()
