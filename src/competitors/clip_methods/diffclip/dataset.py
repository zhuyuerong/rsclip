#!/usr/bin/env python3
"""
DIOR / mini-DIOR dataset utilities tailored for Experiment 7.

This module provides a lightweight PyTorch dataset that reads the DIOR-family
split files, parses VOC-style XML annotations, and produces (image, text)
pairs suitable for CLIP-style contrastive training.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

__all__ = [
    "DIORDataset",
    "MiniDIORDataset",
    "build_train_transform",
    "build_default_transform",
    "MiniDIORSample",
]


@dataclass(frozen=True)
class MiniDIORSample:
    """Structure describing a single sample entry."""

    image_path: str
    class_name: str
    bboxes: Sequence[Tuple[int, int, int, int]]


def build_default_transform(image_size: int = 224) -> Callable[[Image.Image], torch.Tensor]:
    """Evaluation transform matching CLIP statistics."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


def build_train_transform(image_size: int = 224) -> Callable[[Image.Image], torch.Tensor]:
    """Augmented training transform for improved generalisation."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.6, 1.0),
            ratio=(0.75, 1.3333),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


class DIORDataset(Dataset):
    """
    PyTorch dataset for DIOR / mini-DIOR splits.

    The dataset produces (image, text, meta) triples where text is a formatted
    description derived from object annotations. Each unique (image, category)
    combination is treated as an individual training sample.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        text_templates: Optional[Sequence[str]] = None,
        annotation_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
        annotation_type: str = "horizontal",
        image_subdir: Optional[str] = None,
        augment: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.root = root
        self.split = split
        self.text_templates = text_templates or (
            "a satellite photo of {}",
            "an aerial image of {}",
            "a remote sensing picture of {}",
            "a overhead view of {}",
        )

        if augment is None:
            augment = split == "train"
        if transform is None:
            transform = build_train_transform() if augment else build_default_transform()
        self.transform = transform
        self.augment = augment

        if annotation_dir is not None:
            self.annotation_dir = annotation_dir
        else:
            preferred = os.path.join(root, "annotations", annotation_type)
            fallback = os.path.join(root, "annotations")
            self.annotation_dir = preferred if os.path.isdir(preferred) else fallback

        if image_dir is not None:
            self.image_dir = image_dir
        else:
            default_subdir = image_subdir or ("test" if split == "test" else "trainval")
            candidate = os.path.join(root, "images", default_subdir)
            fallback = os.path.join(root, "images")
            self.image_dir = candidate if os.path.isdir(candidate) else fallback

        self.split_file = os.path.join(root, "splits", f"{split}.txt")

        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        ids = self._read_split_ids(self.split_file)
        self.samples: List[MiniDIORSample] = []
        class_set = set()

        for image_id in ids:
            xml_path = self._resolve_annotation_path(image_id)
            img_path = self._resolve_image_path(image_id)
            if xml_path is None or img_path is None:
                continue

            class_to_boxes: Dict[str, List[Tuple[int, int, int, int]]] = {}
            tree = ET.parse(xml_path)
            root_elem = tree.getroot()
            for obj in root_elem.findall("object"):
                name_node = obj.find("name")
                bbox_node = obj.find("bndbox")
                if name_node is None or bbox_node is None:
                    continue
                class_name = name_node.text.strip().lower()
                bbox = (
                    int(bbox_node.find("xmin").text),
                    int(bbox_node.find("ymin").text),
                    int(bbox_node.find("xmax").text),
                    int(bbox_node.find("ymax").text),
                )
                class_to_boxes.setdefault(class_name, []).append(bbox)
                class_set.add(class_name)

            for class_name, boxes in class_to_boxes.items():
                self.samples.append(MiniDIORSample(img_path, class_name, boxes))

        if not self.samples:
            raise RuntimeError(
                f"No samples parsed for split '{split}'. "
                "Please check the dataset integrity."
            )

        self.classes: List[str] = sorted(class_set)

    @staticmethod
    def _read_split_ids(path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        image_tensor = self.transform(image)
        if self.augment and len(self.text_templates) > 1:
            template = random.choice(self.text_templates)
        else:
            template = self.text_templates[0]
        text = template.format(sample.class_name)
        meta = {
            "image_path": sample.image_path,
            "class_name": sample.class_name,
            "bboxes": sample.bboxes,
        }
        return image_tensor, text, meta

    def class_frequencies(self) -> Dict[str, int]:
        freq: Dict[str, int] = {}
        for sample in self.samples:
            freq[sample.class_name] = freq.get(sample.class_name, 0) + 1
        return freq

    def _resolve_annotation_path(self, image_id: str) -> Optional[str]:
        candidates = [
            os.path.join(self.annotation_dir, f"{image_id}.xml"),
            os.path.join(self.root, "annotations", "horizontal", f"{image_id}.xml"),
            os.path.join(self.root, "annotations", "oriented", f"{image_id}.xml"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _resolve_image_path(self, image_id: str) -> Optional[str]:
        extensions = [".jpg", ".png", ".jpeg", ".tif", ".bmp"]
        candidate_dirs = [
            self.image_dir,
            os.path.join(self.root, "images", "trainval"),
            os.path.join(self.root, "images", "test"),
            os.path.join(self.root, "images"),
        ]
        for directory in candidate_dirs:
            if not os.path.isdir(directory):
                continue
            for ext in extensions:
                path = os.path.join(directory, f"{image_id}{ext}")
                if os.path.exists(path):
                    return path
        return None


# Backwards compatibility alias for earlier scripts.
MiniDIORDataset = DIORDataset

