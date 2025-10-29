#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比VV机制和正常机制的Patch特征与文本对齐度

对比指标：
1. Patch特征与标签文本的对齐程度（相似度分布、命中率）
2. Patch位置与bbox的重合程度（IoU、定位准确率）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from PIL import Image
from torchvision import transforms
import xml.etree.ElementTree as ET

# DIOR类别（必须在模块级别定义）
DIOR_CLASSES = [
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
    "chimney", "dam", "expressway-service-area", "expressway-toll-station",
    "golffield", "groundtrackfield", "harbor", "overpass", "ship",
    "stadium", "storagetank", "tenniscourt", "trainstation", "vehicle",
    "windmill"
]

# 导入实验4的模块
import sys
import importlib.util

# 直接导入clip_surgery模块（绕过__init__.py）
clip_surgery_path = Path(__file__).parent / "models" / "clip_surgery.py"
spec = importlib.util.spec_from_file_location("clip_surgery", clip_surgery_path)
clip_surgery_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clip_surgery_module)
CLIPSurgery = clip_surgery_module.CLIPSurgery
CLIPSurgeryWrapper = clip_surgery_module.CLIPSurgeryWrapper

# 导入其他模块
from experiment4.config import Config

# 创建简化的DIOR数据集加载器（如果不存在）
class SimpleDiorDataset:
    """简化的DIOR数据集加载器"""
    def __init__(self, root_dir, split='val'):
        self.root_dir = Path(root_dir)
        self.split = split
        
        # 加载图像路径和标注
        self.samples = []
        
        # 尝试多种可能的路径结构
        possible_image_dirs = [
            self.root_dir / "images" / split,
            self.root_dir / "images" / "trainval" if split == "val" else self.root_dir / "images" / split,
            self.root_dir / "images",
            self.root_dir / split,
        ]
        
        images_dir = None
        for img_dir in possible_image_dirs:
            if img_dir.exists():
                images_dir = img_dir
                break
        
        if images_dir is None:
            raise FileNotFoundError(f"找不到DIOR图像目录。尝试过的路径: {possible_image_dirs}")
        
        # DIOR的标注可能在子目录中（horizontal或oriented）
        annotations_base = self.root_dir / "annotations"
        if not annotations_base.exists():
            raise FileNotFoundError(f"找不到DIOR标注目录: {annotations_base}")
        
        # 优先使用horizontal目录，如果没有则使用oriented
        annotations_dir = annotations_base / "horizontal"
        if not annotations_dir.exists():
            annotations_dir = annotations_base / "oriented"
        if not annotations_dir.exists():
            # 如果没有子目录，直接使用annotations
            annotations_dir = annotations_base
        
        print(f"  使用图像目录: {images_dir}")
        print(f"  使用标注目录: {annotations_dir}")
        
        # 获取split文件
        split_file = self.root_dir / "splits" / f"{split}.txt"
        if split_file.exists():
            with open(split_file, 'r') as f:
                image_names = [line.strip() for line in f if line.strip()]
            print(f"  从split文件加载: {len(image_names)} 个图像名")
        else:
            # 如果没有split文件，使用所有图像
            image_names = [f.name for f in images_dir.glob("*.jpg")]
            if len(image_names) == 0:
                image_names = [f.name for f in images_dir.glob("*.png")]
            print(f"  从目录直接加载: {len(image_names)} 个图像")
        
        if len(image_names) == 0:
            print(f"  ⚠️ 未找到任何图像文件！")
            return
        
        # 加载每个图像的标注
        valid_count = 0
        skipped_no_image = 0
        skipped_no_xml = 0
        skipped_no_valid_bbox = 0
        
        for img_name in image_names[:1000]:  # 限制前1000个以加快速度
            # split文件中的名字可能是纯数字（如"05863"），需要补零和添加扩展名
            img_name_clean = img_name.strip()
            # 尝试补零到5位（DIOR格式）
            if img_name_clean.isdigit():
                img_name_clean = img_name_clean.zfill(5)
            
            # 处理图像文件名
            base_name = Path(img_name_clean).stem
            if not base_name:
                base_name = img_name_clean
            
            # 尝试多种可能的图像文件路径
            possible_img_paths = [
                images_dir / f"{base_name}.jpg",
                images_dir / f"{base_name}.png",
                images_dir / img_name_clean,
                images_dir / f"{img_name_clean}.jpg",
            ]
            
            img_path = None
            for path in possible_img_paths:
                if path.exists():
                    img_path = path
                    break
            
            if img_path is None:
                skipped_no_image += 1
                continue
            
            # XML文件路径（尝试多种可能的格式）
            possible_xml_bases = [
                base_name.zfill(5) if base_name.isdigit() else base_name,
                base_name.zfill(6) if base_name.isdigit() else base_name,
                base_name,
                img_name_clean.zfill(5) if img_name_clean.isdigit() else img_name_clean,
            ]
            
            xml_path = None
            for xml_base in possible_xml_bases:
                candidate_xml = annotations_dir / f"{xml_base}.xml"
                if candidate_xml.exists():
                    xml_path = candidate_xml
                    break
            
            if xml_path is None:
                skipped_no_xml += 1
                if skipped_no_xml <= 3:  # 只在开始时打印示例
                    print(f"  ⚠️ 未找到XML: base_name={base_name}, img_name_clean={img_name_clean}")
                continue
            
            try:
                # 解析XML
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                boxes = []
                labels = []
                
                for obj in root.findall('object'):
                    name_elem = obj.find('name')
                    if name_elem is None:
                        continue
                    
                    class_name = name_elem.text.strip().lower()
                    # DIOR类别的可能变体
                    class_variants = {
                        'expressway-service-area': ['expressway-service-area', 'expresswayservicearea'],
                        'expressway-toll-station': ['expressway-toll-station', 'expresswaytollstation'],
                        'baseballfield': ['baseballfield', 'baseball field'],
                        'basketballcourt': ['basketballcourt', 'basketball court'],
                        'groundtrackfield': ['groundtrackfield', 'ground track field'],
                        'storagetank': ['storagetank', 'storage tank'],
                        'tenniscourt': ['tenniscourt', 'tennis court'],
                        'trainstation': ['trainstation', 'train station'],
                    }
                    
                    # 标准化类别名
                    normalized_name = class_name
                    for canonical, variants in class_variants.items():
                        if class_name in variants:
                            normalized_name = canonical
                            break
                    
                    if normalized_name not in DIOR_CLASSES:
                        continue
                    
                    class_idx = DIOR_CLASSES.index(normalized_name)
                    
                    bbox = obj.find('bndbox')
                    
                    if bbox is None:
                        continue
                    
                    try:
                        # 获取bbox坐标（DIOR数据集通常是800x800）
                        xmin_elem = bbox.find('xmin')
                        ymin_elem = bbox.find('ymin')
                        xmax_elem = bbox.find('xmax')
                        ymax_elem = bbox.find('ymax')
                        
                        if all(elem is not None for elem in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]):
                            x_min = float(xmin_elem.text) / 800.0  # 归一化到[0,1]
                            y_min = float(ymin_elem.text) / 800.0
                            x_max = float(xmax_elem.text) / 800.0
                            y_max = float(ymax_elem.text) / 800.0
                            
                            # 确保坐标在有效范围内
                            x_min = max(0, min(1, x_min))
                            y_min = max(0, min(1, y_min))
                            x_max = max(0, min(1, x_max))
                            y_max = max(0, min(1, y_max))
                            
                            if x_max > x_min and y_max > y_min:
                                # 转换为[cx, cy, w, h]格式（归一化）
                                cx = (x_min + x_max) / 2
                                cy = (y_min + y_max) / 2
                                w = x_max - x_min
                                h = y_max - y_min
                                
                                boxes.append([cx, cy, w, h])
                                labels.append(class_idx)
                    except (ValueError, AttributeError) as e:
                        continue
                
                if len(boxes) > 0:
                    self.samples.append({
                        'image_path': img_path,
                        'boxes': torch.tensor(boxes, dtype=torch.float32),
                        'labels': torch.tensor(labels, dtype=torch.long)
                    })
                    valid_count += 1
                else:
                    skipped_no_valid_bbox += 1
            except ET.ParseError as e:
                print(f"  ⚠️ XML解析错误: {xml_path} - {e}")
                continue
            except Exception as e:
                if valid_count < 5:  # 只在开始时打印详细错误
                    print(f"  ⚠️ 处理错误 {xml_path}: {e}")
                continue
        
        print(f"  成功加载: {valid_count} 个有效样本")
        if skipped_no_image > 0:
            print(f"  跳过（无图像）: {skipped_no_image} 个")
        if skipped_no_xml > 0:
            print(f"  跳过（无XML）: {skipped_no_xml} 个")
        if skipped_no_valid_bbox > 0:
            print(f"  跳过（无有效bbox）: {skipped_no_valid_bbox} 个")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        target = {
            'boxes': sample['boxes'],
            'labels': sample['labels']
        }
        return image, target



class VVAttention(nn.Module):
    """
    VV自注意力机制：Attention(V, V, V)
    
    核心思想：用V替换Q和K，只使用V进行自注意力计算
    """
    
    def __init__(self, dim, num_heads=8, scale_multiplier=1.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = (self.head_dim ** -0.5) * scale_multiplier
        
        # 只需要一个projection layer（用于V）
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.device = None  # 将在首次forward时设置
        
    def forward(self, query, key=None, value=None, need_weights=False, attn_mask=None):
        """
        兼容MultiheadAttention接口的forward方法
        
        Args:
            query: [N, B, D] (CLIP使用序列优先格式)
            key: 未使用（VV机制中key=query）
            value: 未使用（VV机制中value=query）
            need_weights: 是否返回注意力权重
            attn_mask: 未使用
        
        Returns:
            out: [N, B, D] 或 (out, attn_weights) if need_weights=True
        """
        # CLIP使用序列优先格式 [N, B, D]
        N, B, D = query.shape
        
        # 确保在正确的设备上
        if self.device is None:
            self.device = query.device
            self.to_v = self.to_v.to(self.device)
        
        x = query.permute(1, 0, 2)  # [B, N, D]
        
        # 只计算V（确保在正确设备）
        if x.device != self.device:
            x = x.to(self.device)
        v = self.to_v(x)  # [B, N, D]
        
        # Reshape为多头
        v = v.reshape(B, N, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)  # [B, num_heads, N, head_dim]
        
        # VV注意力：Attention(V, V, V) = softmax(VV^T / scale) V
        # Q = K = V（都来自v）
        q_vv = v
        k_vv = v
        
        # L2归一化（可选，提高稳定性）
        q_vv = F.normalize(q_vv, p=2, dim=-1)
        k_vv = F.normalize(k_vv, p=2, dim=-1)
        
        # 计算注意力
        attn = (q_vv @ k_vv.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力到V
        out = attn @ v  # [B, num_heads, N, head_dim]
        
        # Reshape回原始形状
        out = out.permute(0, 2, 1, 3)  # [B, N, num_heads, head_dim]
        out = out.reshape(B, N, D)
        
        # 转换回序列优先格式 [N, B, D]
        out = out.permute(1, 0, 2)
        
        if need_weights:
            # 返回平均注意力权重 [B, N, N]
            attn_weights = attn.mean(dim=1)  # 平均所有头
            return out, attn_weights
        else:
            return out


class VVTransformerBlock(nn.Module):
    """使用VV注意力的Transformer Block"""
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, scale_multiplier=1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = VVAttention(dim, num_heads, scale_multiplier)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
        
    def forward(self, x):
        # 先应用VV注意力（残差连接）
        x = x + self.attn(self.norm1(x))
        # 再应用MLP（残差连接）
        x = x + self.mlp(self.norm2(x))
        return x


class VVCLIPSurgery(nn.Module):
    """
    使用VV机制的CLIP Surgery模型
    
    在最后几层使用VV注意力替换标准注意力
    """
    
    def __init__(self, clip_model, device="cuda", num_vv_blocks=6):
        super().__init__()
        self.clip_model = clip_model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.num_vv_blocks = num_vv_blocks
        
        # 冻结所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.clip_model.eval()
        
        # 确保模型在正确设备上
        self.clip_model = self.clip_model.to(self.device)
        
        # 替换最后num_vv_blocks层的注意力为VV注意力
        self._replace_with_vv_attention()
    
    def _replace_with_vv_attention(self):
        """替换最后几层的注意力为VV注意力"""
        visual = self.clip_model.visual
        transformer = visual.transformer
        
        if not hasattr(transformer, 'resblocks'):
            print("⚠️ 无法找到transformer.resblocks，跳过VV机制替换")
            return
        
        # 获取每层的参数
        first_block = transformer.resblocks[0]
        original_attn = first_block.attn
        
        # 获取维度信息
        if hasattr(original_attn, 'in_proj_weight'):
            dim = original_attn.in_proj_weight.shape[1]  # 768
            in_proj_weight = original_attn.in_proj_weight  # [3*768, 768]
        elif hasattr(original_attn, 'qkv'):
            dim = original_attn.qkv.weight.shape[1]
        else:
            print("⚠️ 无法识别注意力层结构，跳过VV机制替换")
            return
        
        # 获取head数量
        if hasattr(original_attn, 'num_heads'):
            num_heads = original_attn.num_heads
        else:
            # 默认值
            num_heads = 12
        
        # 只替换最后num_vv_blocks层
        replaced_count = 0
        for i in range(1, min(self.num_vv_blocks + 1, len(transformer.resblocks) + 1)):
            block = transformer.resblocks[-i]
            original_attn = block.attn
            
            # 创建VV注意力
            vv_attn = VVAttention(dim, num_heads, scale_multiplier=1.0).to(self.device)
            
            # 从原始注意力复制权重（只复制V的权重）
            if hasattr(original_attn, 'in_proj_weight'):
                in_proj_weight = original_attn.in_proj_weight.data
                v_weight = in_proj_weight[2*dim:, :]  # 提取V的权重 [768, 768]
                
                # 初始化VV注意力的权重
                with torch.no_grad():
                    vv_attn.to_v.weight.data = v_weight.clone().to(self.device)
            elif hasattr(original_attn, 'qkv'):
                qkv_weight = original_attn.qkv.weight.data
                v_weight = qkv_weight[2*dim:, :]
                with torch.no_grad():
                    vv_attn.to_v.weight.data = v_weight.clone().to(self.device)
            
            # 替换注意力模块
            block.attn = vv_attn
            replaced_count += 1
        
        print(f"  ✓ 已替换最后{replaced_count}层为VV注意力")
    
    @classmethod
    def from_pretrained(cls, clip_model, device="cuda", num_vv_blocks=6):
        """从已有CLIP模型创建VV版本"""
        return cls(clip_model, device, num_vv_blocks)
    
    def encode_image(self, images):
        """
        编码图像，使用VV注意力
        
        Args:
            images: [B, 3, 224, 224]
        
        Returns:
            features: [B, 50, 512] (包含CLS token)
        """
        with torch.no_grad():
            # 确保输入在正确的设备上
            if images.device != self.device:
                images = images.to(self.device)
            
            # 确保输入类型匹配
            if images.dtype != self.clip_model.visual.conv1.weight.dtype:
                images = images.to(self.clip_model.visual.conv1.weight.dtype)
            
            # 获取ViT的patch embeddings
            x = self.clip_model.visual.conv1(images)  # [B, 768, 7, 7]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, 768, 49]
            x = x.permute(0, 2, 1)  # [B, 49, 768]
            
            # 添加CLS token
            x = torch.cat([
                self.clip_model.visual.class_embedding.to(x.dtype).to(self.device) + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=self.device
                ),
                x
            ], dim=1)  # [B, 50, 768]
            
            pos_embed = self.clip_model.visual.positional_embedding.to(x.dtype).to(self.device)
            x = x + pos_embed
            x = self.clip_model.visual.ln_pre(x)
            
            # 通过transformer（最后几层使用VV注意力）
            x = x.permute(1, 0, 2)  # [50, B, 768]
            x = self.clip_model.visual.transformer(x)
            x = x.permute(1, 0, 2)  # [B, 50, 768]
            
            # Layer norm
            x = self.clip_model.visual.ln_post(x)
            
            # 投影到512维
            if hasattr(self.clip_model.visual, 'proj') and self.clip_model.visual.proj is not None:
                B, N, D = x.shape
                x_reshaped = x.reshape(B * N, D)
                proj_weight = self.clip_model.visual.proj.to(self.device)
                x_proj = x_reshaped @ proj_weight
                features = x_proj.reshape(B, N, -1)
            else:
                features = x
        
        return features


class VVCLIPSurgeryWrapper:
    """VV机制的CLIP Surgery包装器"""
    
    def __init__(self, config, num_vv_blocks=6):
        self.config = config
        self.device = config.device
        
        # 先加载正常的CLIP Surgery模型
        normal_model = CLIPSurgery.from_pretrained(
            model_name=config.backbone,
            device=self.device
        )
        
        # 转换为VV版本
        self.model = VVCLIPSurgery.from_pretrained(
            normal_model.clip_model,
            device=self.device,
            num_vv_blocks=num_vv_blocks
        )
        
        # 预计算背景词特征（使用原始模型的文本编码器）
        self.bg_features = self.encode_text(config.background_words)
    
    def encode_image(self, images):
        """编码图像"""
        return self.model.encode_image(images)
    
    def encode_text(self, text_list):
        """编码文本（使用原始CLIP的文本编码器）"""
        with torch.no_grad():
            if isinstance(text_list, list):
                import clip
                text_tokens = clip.tokenize(text_list).to(self.device)
            else:
                text_tokens = text_list
            
            text_features = self.model.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def get_all_features(self, images):
        """获取完整特征（包含CLS token + patches）"""
        return self.model.encode_image(images)
    
    def get_cls_features(self, images):
        """获取CLS token特征"""
        all_features = self.model.encode_image(images)
        return all_features[:, 0, :]
    
    def get_patch_features(self, images):
        """获取patch特征（去掉CLS token）"""
        all_features = self.model.encode_image(images)
        return all_features[:, 1:, :]


def patch_to_image_coordinates(patch_idx, grid_size, image_size=224):
    """将patch索引转换为图像坐标"""
    row = patch_idx // grid_size
    col = patch_idx % grid_size
    
    patch_size = image_size // grid_size
    
    x_min = col * patch_size
    y_min = row * patch_size
    x_max = x_min + patch_size
    y_max = y_min + patch_size
    
    return x_min, y_min, x_max, y_max


def calculate_iou(box1, box2):
    """计算两个bbox的IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        inter_area = 0
    else:
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # 计算并集
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def analyze_patch_text_alignment(patch_features, text_features, gt_bboxes, grid_size, iou_threshold=0.05):
    """
    分析patch特征与文本的对齐度
    
    Args:
        patch_features: [B, N, 512] - patch特征
        text_features: [K, 512] - 文本特征（K个类别）
        gt_bboxes: list - 每个样本的bbox列表，格式: [[class_idx, x_min, y_min, x_max, y_max], ...]
        grid_size: int - patch网格大小
        iou_threshold: float - IoU阈值
    
    Returns:
        results: dict - 包含各种对齐度指标
    """
    B, N, D = patch_features.shape
    K = text_features.shape[0]
    
    # 确保dtype一致（转换为float32）
    patch_features = patch_features.float()
    text_features = text_features.float()
    
    # 归一化
    patch_features_norm = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-8)
    text_features_norm = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-8)
    
    # 计算相似度矩阵 [B, N, K]
    similarity = patch_features_norm @ text_features_norm.t()  # [B, N, K]
    
    # 统计指标
    top1_hit_count = 0
    top5_hit_count = 0
    total_samples = 0
    
    top1_ious = []
    top5_ious = []
    max_similarities = []
    
    for b in range(B):
        if b >= len(gt_bboxes) or len(gt_bboxes[b]) == 0:
            continue
        
        total_samples += 1
        sim_b = similarity[b]  # [N, K]
        
        # 获取该样本的所有真实bbox
        gt_boxes_b = gt_bboxes[b]
        
        # 对每个类别，找到最相似的patch
        best_patch_idx = sim_b.argmax(dim=0).cpu().numpy()  # [K]
        best_sim = sim_b.max(dim=0)[0].cpu().numpy()  # [K]
        
        # 检查每个真实bbox对应的类别
        for class_idx, x_min, y_min, x_max, y_max in gt_boxes_b:
            if class_idx >= K:
                continue
            
            # 获取该类别的top-1 patch
            top_patch_idx = best_patch_idx[class_idx]
            patch_bbox = patch_to_image_coordinates(top_patch_idx, grid_size)
            
            # 转换gt_bbox到像素坐标
            gt_bbox_pixel = (
                int(x_min * 224),
                int(y_min * 224),
                int(x_max * 224),
                int(y_max * 224)
            )
            
            # 计算IoU
            iou = calculate_iou(patch_bbox, gt_bbox_pixel)
            top1_ious.append(iou)
            max_similarities.append(best_sim[class_idx].item())
            
            # 检查是否命中
            if iou >= iou_threshold:
                top1_hit_count += 1
            
            # Top-5 patches
            top5_patches = sim_b[:, class_idx].topk(k=min(5, N))[1].cpu().numpy()
            top5_hit = False
            top5_max_iou = 0
            for patch_idx in top5_patches:
                patch_bbox = patch_to_image_coordinates(patch_idx, grid_size)
                iou = calculate_iou(patch_bbox, gt_bbox_pixel)
                top5_max_iou = max(top5_max_iou, iou)
                if iou >= iou_threshold:
                    top5_hit = True
                    break
            
            if top5_hit:
                top5_hit_count += 1
            top5_ious.append(top5_max_iou)
    
    results = {
        'top1_hit_rate': top1_hit_count / total_samples if total_samples > 0 else 0.0,
        'top5_hit_rate': top5_hit_count / total_samples if total_samples > 0 else 0.0,
        'mean_top1_iou': np.mean(top1_ious) if top1_ious else 0.0,
        'mean_top5_iou': np.mean(top5_ious) if top5_ious else 0.0,
        'mean_max_similarity': np.mean(max_similarities) if max_similarities else 0.0,
        'total_samples': total_samples
    }
    
    return results


def main():
    """主函数"""
    print("="*70)
    print("对比VV机制和正常机制的Patch特征对齐度")
    print("="*70)
    
    config = Config()
    
    # 加载数据集
    print("\n加载DIOR数据集...")
    dior_root = Path("datasets/DIOR")
    if not dior_root.exists():
        alt_root = Path(__file__).parent.parent / "datasets" / "DIOR"
        if alt_root.exists():
            dior_root = alt_root
        else:
            raise FileNotFoundError(f"找不到DIOR数据集: {dior_root}")
    
    val_dataset = SimpleDiorDataset(root_dir=dior_root, split='val')
    print(f"  验证集大小: {len(val_dataset)}")
    
    # 加载文本特征
    print("\n加载文本特征...")
    class_names = DIOR_CLASSES
    normal_model = CLIPSurgeryWrapper(config)
    text_features = normal_model.encode_text(class_names).float().to(config.device)  # [20, 512]
    print(f"  文本特征形状: {text_features.shape}, 设备: {text_features.device}")
    
    # 创建VV模型
    print("\n创建VV机制模型...")
    vv_model = VVCLIPSurgeryWrapper(config, num_vv_blocks=6)
    print(f"  ✓ VV模型创建完成（最后6层使用VV注意力）")
    
    # 分析
    print("\n" + "="*70)
    print("开始分析...")
    print("="*70)
    
    normal_results = []
    vv_results = []
    
    max_samples = 500
    
    # 图像预处理
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for idx in tqdm(range(min(max_samples, len(val_dataset))), desc="处理样本"):
        image, target = val_dataset[idx]
        
        # 预处理图像
        if isinstance(image, Image.Image):
            image_tensor = image_transform(image).unsqueeze(0).to(config.device)
        else:
            image_tensor = image.unsqueeze(0).to(config.device)
        
        # 提取bbox信息
        gt_bboxes = []
        if isinstance(target, dict) and 'boxes' in target and 'labels' in target:
            boxes = target['boxes'].cpu().numpy()  # [N, 4] [cx, cy, w, h] 归一化
            labels = target['labels'].cpu().numpy()  # [N]
            
            for box, label in zip(boxes, labels):
                cx, cy, w, h = box
                x_min = max(0, min(1, cx - w / 2))
                y_min = max(0, min(1, cy - h / 2))
                x_max = max(0, min(1, cx + w / 2))
                y_max = max(0, min(1, cy + h / 2))
                
                gt_bboxes.append([int(label), x_min, y_min, x_max, y_max])
        
        if len(gt_bboxes) == 0:
            continue
        
        # 确保是float32
        if image_tensor.dtype != torch.float32:
            image_tensor = image_tensor.float()
        
        try:
            # 1. 正常机制
            normal_patches = normal_model.get_patch_features(image_tensor)  # [1, 49, 512]
            # 确保在正确的设备上
            normal_patches = normal_patches.to(config.device).float()
            grid_size = int(normal_patches.shape[1] ** 0.5)
            
            normal_result = analyze_patch_text_alignment(
                normal_patches, text_features, [gt_bboxes], grid_size
            )
            normal_results.append(normal_result)
            
            # 2. VV机制
            vv_patches = vv_model.get_patch_features(image_tensor)  # [1, 49, 512]
            # 确保在正确的设备上
            vv_patches = vv_patches.to(config.device).float()
            
            vv_result = analyze_patch_text_alignment(
                vv_patches, text_features, [gt_bboxes], grid_size
            )
            vv_results.append(vv_result)
            
        except Exception as e:
            if len(normal_results) + len(vv_results) < 10:  # 只打印前几个错误
                print(f"\n⚠️ 样本{idx}处理失败: {e}")
                import traceback
                traceback.print_exc()
            continue
    
    # 统计结果
    print("\n" + "="*70)
    print("结果统计")
    print("="*70)
    
    if normal_results:
        normal_top1 = np.mean([r['top1_hit_rate'] for r in normal_results])
        normal_top5 = np.mean([r['top5_hit_rate'] for r in normal_results])
        normal_iou1 = np.mean([r['mean_top1_iou'] for r in normal_results])
        normal_iou5 = np.mean([r['mean_top5_iou'] for r in normal_results])
        normal_sim = np.mean([r['mean_max_similarity'] for r in normal_results])
        
        print(f"\n【正常机制（标准注意力）】:")
        print(f"  Top-1 patch命中率: {normal_top1*100:.2f}%")
        print(f"  Top-5 patch命中率: {normal_top5*100:.2f}%")
        print(f"  平均Top-1 IoU: {normal_iou1:.4f}")
        print(f"  平均Top-5 IoU: {normal_iou5:.4f}")
        print(f"  平均最大相似度: {normal_sim:.6f}")
        print(f"  有效样本数: {normal_results[0]['total_samples']}")
    else:
        print(f"\n【正常机制】: 无有效结果")
        normal_top1 = normal_top5 = normal_iou1 = normal_iou5 = normal_sim = 0.0
    
    if vv_results:
        vv_top1 = np.mean([r['top1_hit_rate'] for r in vv_results])
        vv_top5 = np.mean([r['top5_hit_rate'] for r in vv_results])
        vv_iou1 = np.mean([r['mean_top1_iou'] for r in vv_results])
        vv_iou5 = np.mean([r['mean_top5_iou'] for r in vv_results])
        vv_sim = np.mean([r['mean_max_similarity'] for r in vv_results])
        
        print(f"\n【VV机制（Attention(V,V,V)）】:")
        print(f"  Top-1 patch命中率: {vv_top1*100:.2f}%")
        print(f"  Top-5 patch命中率: {vv_top5*100:.2f}%")
        print(f"  平均Top-1 IoU: {vv_iou1:.4f}")
        print(f"  平均Top-5 IoU: {vv_iou5:.4f}")
        print(f"  平均最大相似度: {vv_sim:.6f}")
        print(f"  有效样本数: {vv_results[0]['total_samples']}")
    else:
        print(f"\n【VV机制】: 无有效结果")
        vv_top1 = vv_top5 = vv_iou1 = vv_iou5 = vv_sim = 0.0
    
    # 对比
    if normal_results and vv_results:
        print(f"\n" + "="*70)
        print("对比分析")
        print("="*70)
        
        print(f"\n{'指标':<25s} {'正常机制':>15s} {'VV机制':>15s} {'变化':>15s}")
        print(f"{'-'*70}")
        print(f"{'Top-1命中率':<25s} {normal_top1*100:>14.2f}% {vv_top1*100:>14.2f}% {(vv_top1-normal_top1)*100:>+14.2f}%")
        print(f"{'Top-5命中率':<25s} {normal_top5*100:>14.2f}% {vv_top5*100:>14.2f}% {(vv_top5-normal_top5)*100:>+14.2f}%")
        print(f"{'平均Top-1 IoU':<25s} {normal_iou1:>15.4f} {vv_iou1:>15.4f} {vv_iou1-normal_iou1:>+15.4f}")
        print(f"{'平均Top-5 IoU':<25s} {normal_iou5:>15.4f} {vv_iou5:>15.4f} {vv_iou5-normal_iou5:>+15.4f}")
        print(f"{'平均最大相似度':<25s} {normal_sim:>15.6f} {vv_sim:>15.6f} {vv_sim-normal_sim:>+15.6f}")
        
        print(f"\n结论:")
        if vv_top1 > normal_top1:
            print(f"  ✅ VV机制在Top-1命中率上更好 ({vv_top1*100:.2f}% vs {normal_top1*100:.2f}%)")
        elif normal_top1 > vv_top1:
            print(f"  ⚠️ 正常机制在Top-1命中率上更好 ({normal_top1*100:.2f}% vs {vv_top1*100:.2f}%)")
        else:
            print(f"  ➡️ 两者Top-1命中率接近")
        
        if vv_iou1 > normal_iou1:
            print(f"  ✅ VV机制在定位精度（IoU）上更好 ({vv_iou1:.4f} vs {normal_iou1:.4f})")
        elif normal_iou1 > vv_iou1:
            print(f"  ⚠️ 正常机制在定位精度（IoU）上更好 ({normal_iou1:.4f} vs {vv_iou1:.4f})")
    
    # 保存结果
    output_dir = Path("experiment4/outputs/diagnosis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_summary = {
        'normal_mechanism': {
            'top1_hit_rate': float(normal_top1),
            'top5_hit_rate': float(normal_top5),
            'mean_top1_iou': float(normal_iou1),
            'mean_top5_iou': float(normal_iou5),
            'mean_max_similarity': float(normal_sim),
            'n_samples': normal_results[0]['total_samples'] if normal_results else 0
        },
        'vv_mechanism': {
            'top1_hit_rate': float(vv_top1),
            'top5_hit_rate': float(vv_top5),
            'mean_top1_iou': float(vv_iou1),
            'mean_top5_iou': float(vv_iou5),
            'mean_max_similarity': float(vv_sim),
            'n_samples': vv_results[0]['total_samples'] if vv_results else 0
        },
        'comparison': {
            'top1_hit_rate_change': float(vv_top1 - normal_top1),
            'top5_hit_rate_change': float(vv_top5 - normal_top5),
            'iou1_change': float(vv_iou1 - normal_iou1),
            'iou5_change': float(vv_iou5 - normal_iou5),
            'similarity_change': float(vv_sim - normal_sim)
        },
        'settings': {
            'num_vv_blocks': 6,
            'iou_threshold': 0.05,
            'max_samples': max_samples
        }
    }
    
    output_file = output_dir / 'vv_vs_normal_alignment.json'
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ 结果已保存: {output_file}")


if __name__ == "__main__":
    main()

