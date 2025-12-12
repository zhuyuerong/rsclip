# -*- coding: utf-8 -*-
"""
DeCLIP数据处理（适配遥感数据）

从external/DeCLIP-main/src/training/data.py提取并适配
注意：已移除所有COCO/LVIS相关数据集类，仅保留训练流程框架
"""
import json
import logging
import os
import random
from dataclasses import dataclass
from multiprocessing import Value
from typing import List, Optional
import numpy as np
from .utils import get_tokenizer
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from .open_clip.transform import FixedSizeCrop, _convert_to_rgb, det_image_transform, get_scale, ResizeLongest
try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None
    logging.warning("pycocotools未安装，COCO相关功能不可用")
try:
    from panopticapi import utils
except ImportError:
    utils = None
    logging.warning("panopticapi未安装，相关功能不可用")
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, CenterCrop
import io
try:
    from petrel_client.client import Client
except:
    Client = None



class ProposalDistillDataset(Dataset):
    """
    提案蒸馏数据集（保留用于GridDistillDataset兼容）
    
    注意：此数据集仍使用COCO格式JSON，但建议迁移到遥感数据格式
    使用rs_data_loader.py中的RSProposalDistillDataset替代
    """
    def __init__(self, input_filename, transforms, image_root,
                 crop_size=224,
                 tokenizer=None, args=None):
        if COCO is None:
            raise ImportError("需要pycocotools来加载COCO格式数据，建议使用rs_data_loader.py加载遥感数据")
        logging.debug(f'Loading coco style data from {input_filename}.')
        self.coco = COCO(input_filename)
        logging.debug('Done loading data.')
        self.transforms = transforms
        self.tokenize = tokenizer
        self.image_root = image_root
        self.image_ids = list(self.coco.imgs.keys())
        self.max_anns = 20
        if not isinstance(crop_size, (tuple, list)):
            crop_size = [crop_size, crop_size]
        self.crop_size = crop_size
        self.args = args
        self.min_size = args.min_size
        self.max_size = args.max_size
        self.ceph_root = args.train_ceph_root
        self.use_ceph = (self.ceph_root != "")
        self.FILE_CLIENT = None
        L = args.det_image_size//args.downsample_factor
        if args.use_vfm == "dino-B-8":  # patch 8
            proxy_resolution = L * 8 
        elif args.use_vfm in ["dinov2-L","dinov2-B"]: # patch 14
            proxy_resolution = L* 14
        elif args.use_vfm in ["sam-B","sam-L","dino-B-16"]: # patch 16
            proxy_resolution = L* 16
        else:
            raise NotImplementedError(f"Proxy type '{args.use_vfm}' is not implemented.")
        self.proxy_transform = det_image_transform(
                proxy_resolution,
                is_train=False,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

    def read_image(self, image_name):
        if self.use_ceph:
            image_path = os.path.join(self.ceph_root, image_name)
            if self.FILE_CLIENT is None:
                self.FILE_CLIENT = Client()
            try:
                img_bytes = self.FILE_CLIENT.get(image_path)
                buff = io.BytesIO(img_bytes)
                image = Image.open(buff)
            except:
                print(f"Cannot load {image_path}", flush=True)
                return None
        else:
            image_path = os.path.join(self.image_root, image_name)
            try:
                image = Image.open(image_path)
            except:
                print(f"Cannot load {image_path}", flush=True)
                return None
        width, height = image.size
        if width < 10 or height < 10:
            print(f"Invalid image, size {image.size}", flush=True)
            return None

        return image

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        if 'file_name' in image_info:
            image_name = image_info['file_name']
        else:
            assert 'coco_url' in image_info
            coco_url = image_info['coco_url'].split('/')
            image_name = os.path.join(coco_url[-2], coco_url[-1])

        old_image = self.read_image(image_name)
        proxy_image=self.proxy_transform(old_image)
        if old_image is None:
            next_id = random.choice(range(self.__len__()))
            return self.__getitem__(next_id)
        img_w, img_h = old_image.width, old_image.height
        new_image = self.transforms[0](old_image)
        scale = get_scale(old_image, new_image)
        anns = self.coco.imgToAnns[image_id]
        boxes_template = torch.zeros(self.max_anns, 4 + 1)    # xyxy s
        texts=[]
        image_crops = torch.zeros(self.max_anns, 3, *self.crop_size)
        indices = list(range(len(anns)))
        random.shuffle(indices)
        num_valid_boxes = 0
        for i, ann_id in enumerate(indices[:self.max_anns]):
            ann = anns[ann_id]
            x, y, w, h = ann['bbox']
            if w*h < (self.min_size ** 2) or w*h > (self.max_size ** 2):
                continue
            num_valid_boxes += 1
            cx, cy = x + w*0.5, y + h*0.5
            x0, y0, x1, y1 = \
                max(cx - w*0.75, 0), max(cy - h*0.75, 0), min(cx + w*0.75, img_w), min(cy + h*0.75, img_h)
            image_crops[i] = self.transforms[1](old_image.crop((x0, y0, x1, y1)))   # image crops
            box_info = torch.tensor([x, y, x + w, y + h, 1.0])    # x, y, x + w, y + h
            boxes_template[i] = box_info

        if num_valid_boxes == 0:
            boxes_template[0] = torch.tensor([0, 0, img_w / 4, img_h / 4, 1.0])    # avoid empty
            image_crops[0] = self.transforms[1](old_image.crop((0, 0, img_w // 4, img_h // 4)))

        _, h, w = new_image.shape

        boxes_template[:, :4] *= scale
        boxes_template[:, [0, 2]] /= w
        boxes_template[:, [1, 3]] /= h
        return new_image, boxes_template, image_crops,proxy_image
    


class GridDistillDataset(Dataset):
    """
    网格蒸馏数据集
    
    注意：仍支持COCO格式JSON（通过文件名判断），但建议迁移到遥感数据格式
    """
    def __init__(self,
                 input_filename,
                 transforms,
                 image_root,
                 max_split=16,
                 crop_size=224,
                 pre_transforms=False,
                 ceph_root="",
                 args=None):
        if os.path.basename(input_filename) in ['lvis_v1_train.json', 'instances_train2017.json']:
            # coco style distillation
            if COCO is None:
                raise ImportError("需要pycocotools来加载COCO格式数据，建议使用rs_data_loader.py加载遥感数据")
            logging.debug(f'Loading coco style data from {input_filename}.')
            self.coco = COCO(input_filename)
            logging.debug('Done loading data.')
            image_ids = list(self.coco.imgs.keys())
            self.style="coco"
        elif os.path.basename(input_filename) in ['chat.json','mixed_data.json','llava_v1_5_mix624k.json']:
            # llava style distillation
            with open(input_filename, 'r') as file:
                data = json.load(file)
            image_ids = [item["image"] for item in data]
            self.style="llava"
        else:
            raise ValueError(f"Unsupported file format or style for {input_filename}.")
        self._init_choices(max_split)
        self.transforms = transforms
        self.image_root = image_root
        self.args = args
        train_ratio = args.train_ratio
        if train_ratio < 1.0:
            num_images = int(len(image_ids) * train_ratio)
            random.shuffle(image_ids)
            image_ids = image_ids[:num_images]
        self.image_ids = image_ids
        self.max_anns = args.max_boxes
        if not isinstance(crop_size, (tuple, list)):
            crop_size = [crop_size, crop_size]
        self.crop_size = crop_size
        self._init_boxes()
        self.ceph_root = ceph_root
        self.use_ceph = (ceph_root != "")
        self.FILE_CLIENT = None
        L = args.det_image_size//args.downsample_factor
        if args.use_vfm:
            if args.use_vfm == "dino-B-8":  # patch 8
                proxy_resolution = L * 8 
            elif args.use_vfm in ["dinov2-L","dinov2-B"]: # patch 14
                proxy_resolution = L* 14
            elif args.use_vfm in ["sam-B","sam-L","dino-B-16"]: # patch 16
                proxy_resolution = L* 16
            else:
                raise NotImplementedError(f"Proxy type '{args.use_vfm}' is not implemented.")
            self.proxy_transform = det_image_transform(
                    proxy_resolution,
                    is_train=False,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
        else:
            self.proxy_transform=None

    def read_image(self, image_name):
        if self.use_ceph:
            image_path = os.path.join(self.ceph_root, image_name)
            if self.FILE_CLIENT is None:
                self.FILE_CLIENT = Client()
            try:
                img_bytes = self.FILE_CLIENT.get(image_path)
                buff = io.BytesIO(img_bytes)
                image = Image.open(buff)
            except:
                print(f"Cannot load {image_path}", flush=True)
                return None
        else:
            image_path = os.path.join(self.image_root, image_name)
            try:
                image = Image.open(image_path)
            except:
                print(f"Cannot load {image_path}", flush=True)
                return None
        width, height = image.size
        if width < 10 or height < 10:
            print(f"Invalid image, size {image.size}", flush=True)
            return None

        return image

    def _init_choices(self, max_split):
        self.choices = []
        for i in range(1, max_split + 1):
            for j in range(1, max_split + 1):
                self.choices.append((i, j))

    def _init_boxes(self):
        self.boxes = []
        for i, j in self.choices:
            x0 = (i - 1) / i
            y0 = (j - 1) / j
            x1 = 1.0
            y1 = 1.0
            self.boxes.append([x0, y0, x1, y1])

    def __len__(self):
        return len(self.image_ids)

    def _load_target(self, id: int):
        if self.style=="coco":
            return self.coco.loadAnns(self.coco.getAnnIds(id))
        else:
            return []

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        if self.style=="coco":
            image_info = self.coco.imgs[image_id]
            if 'file_name' in image_info:
                image_name = image_info['file_name']
            else:
                assert 'coco_url' in image_info
                coco_url = image_info['coco_url'].split('/')
                image_name = os.path.join(coco_url[-2], coco_url[-1])
        else:
            image_name = image_id

        old_image = self.read_image(image_name)
        if old_image is None:
            next_id = random.choice(range(self.__len__()))
            return self.__getitem__(next_id)
        img_w, img_h = old_image.width, old_image.height
        new_image = self.transforms[0](old_image)
        scale = get_scale(old_image, new_image)
        anns = self._load_target(image_id)
        boxes_template = torch.zeros(self.max_anns, 4 + 1)    # xyxy s
        image_crops = torch.zeros(self.max_anns, 3, *self.crop_size)
        indices = list(range(len(anns)))
        random.shuffle(indices)
        num_valid_boxes = 0
        for i, ann_id in enumerate(indices[:self.max_anns]):
            ann = anns[ann_id]
            x, y, w, h = ann['bbox']
            if w*h < (self.args.min_size ** 2) or w*h > (self.args.max_size ** 2):
                continue
            num_valid_boxes += 1
            cx, cy = x + w*0.5, y + h*0.5
            x0, y0, x1, y1 = \
                max(cx - w*0.75, 0), max(cy - h*0.75, 0), min(cx + w*0.75, img_w), min(cy + h*0.75, img_h)
            image_crops[i] = self.transforms[1](old_image.crop((x0, y0, x1, y1)))   # image crops
            box_info = torch.tensor([x, y, x + w, y + h, 1.0])    # x, y, x + w, y + h
            boxes_template[i] = box_info

        if num_valid_boxes == 0:
            boxes_template[0] = torch.tensor([0, 0, img_w / 4, img_h / 4, 1.0])    # avoid empty
            image_crops[0] = self.transforms[1](old_image.crop((0, 0, img_w // 4, img_h // 4)))

        _, h, w = new_image.shape

        boxes_template[:, :4] *= scale
        boxes_template[:, [0, 2]] /= w
        boxes_template[:, [1, 3]] /= h

        choice_idx = random.choice(range(len(self.choices)))
        i, j = self.choices[choice_idx]
        x0, y0, x1, y1 = self.boxes[choice_idx]
        grid_crop = new_image[:, int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        grid_crop = self.transforms[1](Image.fromarray((grid_crop.permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
        if self.proxy_transform:
            proxy_image = self.proxy_transform(old_image)
        else:
            proxy_image = new_image
        return new_image, boxes_template, image_crops, grid_crop, proxy_image


def get_proposal_distill_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    """
    获取提案蒸馏数据集
    
    注意：此函数仍使用COCO格式，建议使用rs_data_loader.py中的get_rs_proposal_distill_dataset
    """
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    if args.image_crop_size>0 :
        image_crop_size=args.image_crop_size
    else:
        if args.model=="EVA02-CLIP-B-16" or args.model=="ViT-B-16" or args.model=="ViT-L-14":
            image_crop_size=224
        elif args.model=="siglip-so400m-patch14-384":
            image_crop_size=384 
        else:
            image_crop_size=336 # ViT-L-14-336 & EVA02-CLIP-L-14-336
    dataset = ProposalDistillDataset(
        input_filename,
        preprocess_fn,
        image_root=args.train_image_root if is_train else args.val_image_root,
        crop_size=image_crop_size,
        tokenizer=tokenizer,
        args=args
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_grid_distill_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    """
    获取网格蒸馏数据集
    
    注意：仍支持COCO格式JSON，但建议迁移到遥感数据格式
    """
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    if args.image_crop_size>0 :
        image_crop_size=args.image_crop_size
    else:
        if args.model=="EVA02-CLIP-B-16" or args.model=="ViT-B-16" or args.model=="ViT-L-14":
            image_crop_size=224
        elif args.model=="siglip-so400m-patch14-384":
            image_crop_size=384 
        else:
            image_crop_size=336 # ViT-L-14-336 & EVA02-CLIP-L-14-336
    dataset = GridDistillDataset(
        input_filename,
        preprocess_fn,
        image_root=args.train_image_root if is_train else args.val_image_root,
        max_split=args.max_split,
        crop_size=image_crop_size,
        pre_transforms=args.pre_transforms,
        ceph_root=args.train_ceph_root if is_train else args.val_ceph_root,
        args=args
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_region_clip_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    """
    获取RegionCLIP数据集
    
    注意：此函数仍使用COCO格式，建议迁移到遥感数据格式
    """
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    if COCO is None:
        raise ImportError("需要pycocotools来加载COCO格式数据，建议使用rs_data_loader.py加载遥感数据")
    # 注意：COCORegionCLIPDataset已移除，这里保留函数签名但会报错
    # 建议使用rs_data_loader.py中的遥感数据加载器
    raise NotImplementedError(
        "COCORegionCLIPDataset已移除。"
        "请使用rs_data_loader.py中的get_rs_proposal_distill_dataset或迁移到遥感数据格式"
    )


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_fn(data_path, dataset_type):
    """
    获取数据集函数（适配遥感数据）
    
    注意：已移除COCO/LVIS相关数据集，仅支持遥感数据
    """
    if dataset_type == 'proposals_distill':
        # 使用遥感数据加载器
        from .rs_data_loader import get_rs_proposal_distill_dataset
        return get_rs_proposal_distill_dataset
    elif dataset_type == 'grid_distill':
        # GridDistillDataset已适配遥感数据（通过文件格式判断）
        return get_grid_distill_dataset
    elif dataset_type == 'region_clip':
        return get_region_clip_dataset
    else:
        raise ValueError(
            f"Unsupported dataset type: {dataset_type}. "
            "支持的类型: proposals_distill, grid_distill, region_clip. "
            "COCO/LVIS相关数据集已移除，请使用遥感数据格式（DIOR/DOTA/LAE-1M）"
        )


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, dataset_type=args.test_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    return data
