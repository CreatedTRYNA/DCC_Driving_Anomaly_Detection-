#!/usr/bin/env python

import argparse
from typing import Dict

import torch
import torch.distributed as dist
import spatial_transforms
import torchvision
from dataset import DAD
from temporal_transforms import TemporalSequentialCrop
import random
import numpy as np


def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_spatial_temporal_transform(args: argparse.Namespace):
    use_tencrops = False
    args.scales = [args.initial_scales]
    for i in range(1, args.n_scales):
        args.scales.append(args.scales[-1] * args.scale_step)
    assert args.train_crop in ['random', 'corner', 'center']
    if args.train_crop == 'random':
        crop_method = spatial_transforms.MultiScaleRandomCrop(args.scales, args.sample_size)
    elif args.train_crop == 'corner':
        crop_method = spatial_transforms.MultiScaleCornerCrop(args.scales, args.sample_size)
    elif args.train_crop == 'center':
        crop_method = spatial_transforms.MultiScaleCornerCrop(args.scales, args.sample_size, crop_positions=['c'])
    before_crop_duration = int(args.sample_duration * args.downsample)

    temporal_transform = TemporalSequentialCrop(before_crop_duration, args.downsample)

    if args.view == 'front_depth' or args.view == 'front_IR':
        transform_list = [spatial_transforms.Scale((args.sample_size, args.sample_size)),
            crop_method,
            spatial_transforms.RandomRotate(),
            spatial_transforms.SaltImage(),
            spatial_transforms.Dropout(),
            spatial_transforms.ToTensor(args.norm_value),
            spatial_transforms.Normalize([0], [1])]

    elif args.view == 'top_depth' or args.view == 'top_IR':
        if args.view == 'top_depth':
            transform_list = [
                spatial_transforms.RandomHorizontalFlip(),
                spatial_transforms.Scale(args.sample_size),
                spatial_transforms.CenterCrop(args.sample_size),
                # spatial_transforms.Hog_descriptor(),
                spatial_transforms.RandomRotate(),
                spatial_transforms.SaltImage(),
                spatial_transforms.Dropout(),
                spatial_transforms.ToTensor(args.norm_value),
                spatial_transforms.Normalize([0], [1])
            ]
        else:
            transform_list = [
                # spatial_transforms.RandomHorizontalFlip(),
                spatial_transforms.Scale(args.sample_size),
                spatial_transforms.CenterCrop(args.sample_size),

                spatial_transforms.RandomRotate(),
                spatial_transforms.SaltImage(),
                spatial_transforms.Dropout(),
                spatial_transforms.ToTensor(args.norm_value),
                spatial_transforms.Normalize([0], [1])
            ]

    else:
        transform_list = [spatial_transforms.RandomHorizontalFlip(),
            spatial_transforms.Scale(args.sample_size),
            spatial_transforms.CenterCrop(args.sample_size),

            spatial_transforms.RandomRotate(),
            spatial_transforms.SaltImage(),
            spatial_transforms.Dropout(),
            spatial_transforms.ToTensor(args.norm_value),
            spatial_transforms.Normalize([0], [1])]


    if use_tencrops:
        transform_list.append(spatial_transforms.TenCrop(size=112))

    spatial_transform = spatial_transforms.Compose(transform_list)
    return temporal_transform, spatial_transform, before_crop_duration


def create_train_loader(args: argparse.Namespace,
                        type: str = "normal") -> torch.utils.data.DataLoader:
    temporal_transform, spatial_transform, before_crop_duration = get_spatial_temporal_transform(args)
    training_data = DAD(root_path=args.root_path,
                        subset='train',
                        view=args.view,
                        sample_duration=before_crop_duration,
                        type=type,
                        spatial_transform=spatial_transform,
                        temporal_transform=temporal_transform
                        )

    if type == "normal":
        batch_size = args.n_train_batch_size
    else:
        batch_size = args.a_train_batch_size

    train_sampler = torch.utils.data.distributed.DistributedSampler(training_data)
    loader = torch.utils.data.DataLoader(
        training_data, sampler=train_sampler, batch_size=batch_size,
        num_workers=args.n_threads, pin_memory=True, drop_last=True
    )

    return loader, training_data

def create_other_view_train_loader(args: argparse.Namespace,
                        type: str = "normal") -> torch.utils.data.DataLoader:
    temporal_transform, spatial_transform, before_crop_duration = get_spatial_temporal_transform(args)
    training_data = DAD(root_path=args.root_path,
                        subset='train',
                        view=args.view_2,
                        sample_duration=before_crop_duration,
                        type=type,
                        spatial_transform=spatial_transform,
                        temporal_transform=temporal_transform
                        )

    if type == "normal":
        batch_size = args.n_train_batch_size
    else:
        batch_size = args.a_train_batch_size

    train_sampler = torch.utils.data.distributed.DistributedSampler(training_data)
    loader = torch.utils.data.DataLoader(
        training_data, sampler=train_sampler, batch_size=batch_size,
        num_workers=args.n_threads, pin_memory=True, drop_last=True
    )

    return loader, training_data

def create_multi_train_loader(args: argparse.Namespace, rank, world_size, type="normal") -> torch.utils.data.DataLoader:
    temporal_transform, spatial_transform, before_crop_duration = get_spatial_temporal_transform(args)

    view_list = args.view.split(",")
    train_dataloader = None
    train_data = None
    for view in view_list:
        training_data = DAD(root_path=args.root_path,
                            subset='train',
                            view=view,
                            sample_duration=before_crop_duration,
                            type=type,
                            spatial_transform=spatial_transform,
                            temporal_transform=temporal_transform
                            )
        if train_data is None:
            train_data = training_data
        else:
            train_data = train_data + training_data

        if type == "normal":
            batch_size = args.n_train_batch_size
        else:
            batch_size = args.a_train_batch_size

        assert batch_size % world_size == 0
        batch_size_per_gpu = batch_size // world_size

        train_sampler = torch.utils.data.distributed.DistributedSampler(training_data)

        loader = torch.utils.data.DataLoader(
            training_data, sampler=train_sampler, batch_size=batch_size_per_gpu,
            num_workers=args.n_threads, pin_memory=True, drop_last=True,
        )

        if train_dataloader is None:
            train_dataloader = loader
        else:
            train_dataloader = zip(train_dataloader, loader)

    return train_dataloader, train_data


def create_val_loader(args: argparse.Namespace) -> torch.utils.data.Dataset:
    val_spatial_transform = spatial_transforms.Compose([
        spatial_transforms.Scale(args.sample_size),
        spatial_transforms.CenterCrop(args.sample_size),
        spatial_transforms.ToTensor(args.norm_value),
        spatial_transforms.Normalize([0], [1])
    ])
    validation_data = DAD(root_path=args.root_path,
                          subset='validation',
                          view=args.view,
                          sample_duration=args.sample_duration,
                          type=None,
                          spatial_transform=val_spatial_transform,
                          )

    validation_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=True,
    )

    return validation_loader, validation_data

def create_other_view_val_loader(args: argparse.Namespace) -> torch.utils.data.Dataset:
    val_spatial_transform = spatial_transforms.Compose([
        spatial_transforms.Scale(args.sample_size),
        spatial_transforms.CenterCrop(args.sample_size),
        spatial_transforms.ToTensor(args.norm_value),
        spatial_transforms.Normalize([0], [1])
    ])
    validation_data = DAD(root_path=args.root_path,
                          subset='validation',
                          view=args.view_2,
                          sample_duration=args.sample_duration,
                          type=None,
                          spatial_transform=val_spatial_transform,
                          )

    validation_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.n_threads,
        pin_memory=True,
    )

    return validation_loader, validation_data