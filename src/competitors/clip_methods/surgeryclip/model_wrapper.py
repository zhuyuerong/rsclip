# -*- coding: utf-8 -*-
"""
CLIP Surgery模型包装器

实现统一接口，适配遥感数据
"""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
from pathlib import Path
import sys
import os

# 添加本地路径以优先使用本地实现
local_clip_path = os.path.join(os.path.dirname(__file__))
if local_clip_path not in sys.path:
    sys.path.insert(0, local_clip_path)

from .clip_model import CLIP
from .clip_surgery_model import CLIPSurgery
from .build_model import build_model
from ..base_interface import BaseCLIPMethod

# CAL模块导入（可选，如果不存在则不影响）
try:
    from .cal_config import CALConfig, NegativeSampleGenerator
    from .cal_modules import CALFeatureSpace, CALSimilaritySpace, ExperimentTracker
    CAL_AVAILABLE = True
except ImportError:
    CAL_AVAILABLE = False
    CALConfig = None
    NegativeSampleGenerator = None
    CALFeatureSpace = None
    CALSimilaritySpace = None
    ExperimentTracker = None


class SurgeryCLIPWrapper(BaseCLIPMethod):
    """
    CLIP Surgery包装器，实现统一接口
    """
    
    def __init__(self, 
                 model_name: str = "surgeryclip",           # "clip" 或 "surgeryclip"
                 checkpoint_path: str = None,               # 权重文件路径
                 device: str = "cuda", 
                 
                 # 🔥 改为字符串，包含不同策略
                 use_surgery_single: str = "empty",   # "none", "empty", "background", "scene", "prompt_template", "all_classes"
                 use_surgery_multi: bool = True,       # 保持bool
                 
                 # 🔥 CAL实验配置（可选，默认不启用）
                 cal_config: Optional['CALConfig'] = None
                 ):
        """
        Args:
            model_name: 模型架构
                - "clip": 原始CLIP（无VV注意力）
                - "surgeryclip": Surgery CLIP（有VV注意力）
            
            checkpoint_path: 权重文件路径
                - "checkpoints/ViT-B-32.pt": OpenAI CLIP权重
                - "checkpoints/RemoteCLIP-ViT-B-32.pt": RemoteCLIP权重
                - 或其他权重文件
            
            use_surgery_single: 单类别时的Surgery策略
                - "none": 不用Surgery，直接余弦相似度
                - "empty": Surgery + 空字符串redundant（原始方法）
                - "background": Surgery + 添加background类（转多类别）
                - "scene": Surgery + 通用场景描述
                - "prompt_template": Surgery + prompt模板差异
                - "all_classes": Surgery + 其他类别平均
            
            use_surgery_multi: 多类别时是否用Surgery（保持bool）
                - True: 使用Surgery（自动计算redundant）
                - False: 直接余弦相似度
        """
        super().__init__(model_name, device)
        self.checkpoint_path = checkpoint_path
        
        self.use_surgery_single = use_surgery_single
        self.use_surgery_multi = use_surgery_multi
        
        self.model = None
        self.preprocess = None
        
        # DIOR所有类别（必须在CAL初始化之前定义）
        self.all_classes = [
            'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
            'chimney', 'dam', 'Expressway-Service-area', 'Expressway-toll-station',
            'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
            'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'
        ]
        
        # 🔥 CAL模块初始化（可插拔）
        self.cal_config = cal_config
        self.negative_generator = None
        self.cal_feature_space = None
        self.cal_similarity_space = None
        self.experiment_tracker = None
        
        if CAL_AVAILABLE and cal_config is not None and cal_config.enable_cal:
            self.negative_generator = NegativeSampleGenerator(cal_config, self.all_classes)
            self.cal_feature_space = CALFeatureSpace(cal_config)
            self.cal_similarity_space = CALSimilaritySpace(cal_config)
            self.experiment_tracker = ExperimentTracker()
            print(f"✅ CAL已启用: {cal_config.get_experiment_id()}")
        elif cal_config is not None and cal_config.enable_cal:
            print("⚠️  CAL模块未找到，CAL功能将被禁用")

    def load_model(self, checkpoint_path: Optional[str] = None):
        """加载模型"""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        
        if checkpoint_path is None:
            raise ValueError("必须提供 checkpoint_path")
        
        # 🔥 统一使用 build_model，根据 model_name 决定架构
        from .build_model import build_model
        
        self.model, self.preprocess = build_model(
            model_name=self.model_name,  # "clip" 或 "surgeryclip"
            checkpoint_path=checkpoint_path,
            device=self.device
        )
    
    def _find_checkpoint_in_checkpoints_dir(self) -> Optional[str]:
        """在checkpoints目录中查找匹配的权重文件"""
        import os
        # 获取项目根目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
        checkpoints_dir = os.path.join(project_root, "checkpoints")
        
        if not os.path.exists(checkpoints_dir):
            return None
        
        # 模型名称到checkpoint文件的映射
        model_to_checkpoint = {
            "ViT-B/32": "RemoteCLIP-ViT-B-32.pt",  # 优先使用RemoteCLIP权重
            "ViT-B/16": "RemoteCLIP-ViT-B-16.pt",
            "ViT-L/14": "RemoteCLIP-ViT-L-14.pt",
            "RN50": "RemoteCLIP-RN50.pt",
            # 原始CLIP权重作为备选
            "CLIP-ViT-B/32": "ViT-B-32.pt",
            "CLIP-ViT-B/16": "ViT-B-16.pt", 
            "CLIP-ViT-L/14": "ViT-L-14.pt",
        }
        
        checkpoint_name = model_to_checkpoint.get(self.model_name)
        if checkpoint_name:
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_name)
            if os.path.exists(checkpoint_path):
                return checkpoint_path
        
        # 如果没有精确匹配，尝试查找包含模型名称的文件
        for filename in os.listdir(checkpoints_dir):
            if filename.endswith('.pt'):
                # 检查文件名是否包含模型名称的关键部分
                model_key = self.model_name.replace('/', '-').replace('ViT-', 'ViT-')
                if model_key in filename or self.model_name.replace('/', '') in filename:
                    return os.path.join(checkpoints_dir, filename)
        
        return None
    
    def encode_image(self, image: Union[torch.Tensor, Image.Image, np.ndarray]) -> torch.Tensor:
        """编码图像"""
        if self.model is None:
            self.load_model()
        
        # 预处理
        if isinstance(image, Image.Image):
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
            image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device) if not image.is_cuda else image
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor)
        
        return image_features
    
    def encode_text(self, text: Union[str, List[str]], use_prompt_ensemble: bool = True) -> torch.Tensor:
        """编码文本
        
        Args:
            text: 文本或文本列表
            use_prompt_ensemble: 是否使用prompt ensemble（CLIP Surgery的标准方式）
        """
        if self.model is None:
            self.load_model()
        
        if isinstance(text, str):
            text = [text]
        
        if use_prompt_ensemble:
            # 使用CLIP Surgery的prompt ensemble方式（标准方式）
            from .clip import encode_text_with_prompt_ensemble
            with torch.no_grad():
                text_features = encode_text_with_prompt_ensemble(
                    self.model, text, self.device
                )
        else:
            # 简单方式（用于兼容）
            from .clip import tokenize
            text_tokens = tokenize(text).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
            # 归一化
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def compute_similarity(self, image: Union[torch.Tensor, Image.Image, np.ndarray],
                          text: Union[str, List[str]]) -> torch.Tensor:
        """计算相似度"""
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        
        # 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 计算相似度
        similarity = (image_features @ text_features.T) * 100.0  # CLIP的logit_scale
        
        return similarity
    
    def generate_heatmap(self, image: Union[torch.Tensor, Image.Image, np.ndarray],
                        text: Union[str, List[str]],
                        return_features: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """生成热图"""
        if self.model is None:
            self.load_model()
        
        # 预处理图像
        if isinstance(image, Image.Image):
            image_pil = image
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        elif isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
            image_tensor = self.preprocess(image_pil).unsqueeze(0).to(self.device)
        else:
            image_tensor = image.to(self.device) if not image.is_cuda else image
            image_pil = None
        
        # 获取图像尺寸
        if isinstance(image_pil, Image.Image):
            target_h, target_w = image_pil.size[1], image_pil.size[0]
        else:
            target_h, target_w = 224, 224
        
        try:
            with torch.no_grad():
                # 获取图像特征
                image_features_all = self.model.encode_image(image_tensor)
                image_features_all = image_features_all / image_features_all.norm(dim=-1, keepdim=True)
                
                # 使用CLIP Surgery的prompt ensemble方式编码文本
                from .clip import encode_text_with_prompt_ensemble, clip_feature_surgery
                text_features = encode_text_with_prompt_ensemble(self.model, text, self.device)
                
                # 🔥 根据类别数量选择对应的配置
                is_single_class = (len(text) == 1)
                
                # 🔍 关键检查点：确认走了哪个分支
                print(f"\n{'='*60}")
                print(f"generate_heatmap 调用:")
                print(f"  text: {text}")
                print(f"  use_surgery_single: {self.use_surgery_single}")
                print(f"  is_single_class: {is_single_class}")
                
                if is_single_class:
                    # ============ 单类别分支 ============
                    
                    if self.use_surgery_single == "none":
                        # 策略0: 不用Surgery
                        print(f"  ✅ 走分支: none (直接余弦)")
                        print("🔍 单类别: 直接余弦相似度")
                        text_features = encode_text_with_prompt_ensemble(self.model, text, self.device)
                        similarity_maps = image_features_all @ text_features.t()
                    
                    elif self.use_surgery_single == "empty":
                        # 策略1: 空字符串（原始）
                        print(f"  ✅ 走分支: empty (空字符串)")
                        print("🔍 单类别Surgery: 空字符串redundant")
                        text_features = encode_text_with_prompt_ensemble(self.model, text, self.device)
                        redundant_features = encode_text_with_prompt_ensemble(
                            self.model, [""], self.device
                        )
                        similarity_maps = clip_feature_surgery(
                            image_features_all, text_features, redundant_features
                        )
                    
                    elif self.use_surgery_single == "background":
                        # 策略2: 添加background类（转多类别）
                        print(f"  ✅ 走分支: background (添加背景类)")
                        print("🔍 单类别Surgery: 添加background类（转多类别）")
                        extended_text = text + ["background"]
                        extended_text_features = encode_text_with_prompt_ensemble(
                            self.model, extended_text, self.device
                        )
                        similarity_maps = clip_feature_surgery(
                            image_features_all, extended_text_features, redundant_feats=None
                        )
                        # 只取目标类别的相似度
                        similarity_maps = similarity_maps[:, :, 0:1]
                    
                    elif self.use_surgery_single == "scene":
                        # 策略3: 通用场景描述
                        print(f"  ✅ 走分支: scene (场景描述)")
                        print("🔍 单类别Surgery: 通用场景描述")
                        text_features = encode_text_with_prompt_ensemble(self.model, text, self.device)
                        
                        scene_descriptions = [
                            "an aerial photograph",
                            "a satellite image",
                            "remote sensing imagery"
                        ]
                        scene_features = encode_text_with_prompt_ensemble(
                            self.model, scene_descriptions, self.device
                        )
                        # 取平均作为redundant
                        redundant_features = scene_features.mean(dim=0, keepdim=True)
                        
                        similarity_maps = clip_feature_surgery(
                            image_features_all, text_features, redundant_features
                        )
                    
                    elif self.use_surgery_single == "prompt_template":
                        # 策略4: Prompt模板差异
                        print(f"  ✅ 走分支: prompt_template (模板差异)")
                        print("🔍 单类别Surgery: Prompt模板差异")
                        
                        target_class = text[0]
                        # 完整prompt
                        full_prompt = [f"an aerial photo of {target_class}"]
                        text_features = encode_text_with_prompt_ensemble(
                            self.model, full_prompt, self.device
                        )
                        
                        # 只有模板
                        template_prompt = ["an aerial photo of"]
                        redundant_features = encode_text_with_prompt_ensemble(
                            self.model, template_prompt, self.device
                        )
                        
                        similarity_maps = clip_feature_surgery(
                            image_features_all, text_features, redundant_features
                        )
                    
                    elif self.use_surgery_single == "all_classes":
                        # 策略5: 其他类别平均
                        print(f"  ✅ 走分支: all_classes (其他类别平均)")
                        print("🔍 单类别Surgery: 其他类别平均")
                        
                        target_class = text[0]
                        text_features = encode_text_with_prompt_ensemble(self.model, text, self.device)
                        
                        # 其他类别
                        other_classes = [c for c in self.all_classes if c != target_class]
                        other_features = encode_text_with_prompt_ensemble(
                            self.model, other_classes, self.device
                        )
                        # 平均作为redundant
                        redundant_features = other_features.mean(dim=0, keepdim=True)
                        
                        similarity_maps = clip_feature_surgery(
                            image_features_all, text_features, redundant_features
                        )
                    
                    else:
                        raise ValueError(f"未知的use_surgery_single: {self.use_surgery_single}")
                
                else:  # len(text) > 1
                    # ============ 多类别分支 ============
                    text_features = encode_text_with_prompt_ensemble(self.model, text, self.device)
                    
                    if self.use_surgery_multi:
                        print("🔍 多类别: Surgery + 自动计算redundant")
                        similarity_maps = clip_feature_surgery(
                            image_features_all, text_features, redundant_feats=None
                        )
                    else:
                        print("🔍 多类别: 直接余弦相似度")
                        similarity_maps = image_features_all @ text_features.t()
                
                # 统计
                print(f"   相似度: min={similarity_maps.min():.6f}, max={similarity_maps.max():.6f}, std={similarity_maps.std():.6f}")
                
                # 排除class token，只保留patch tokens的相似度
                similarity_maps = similarity_maps[:, 1:, :]  # [batch, num_patches, num_texts]
                
                # 🔍 关键检查点：归一化前的similarity_maps
                print(f"  归一化前 similarity_maps: min={similarity_maps.min():.6f}, max={similarity_maps.max():.6f}, std={similarity_maps.std():.6f}")
                
                # 🔥 CAL模块：相似度空间操作（可插拔）
                if (self.cal_config is not None and 
                    self.cal_config.enable_cal and 
                    self.cal_config.cal_space in ['similarity', 'both'] and
                    self.cal_similarity_space is not None):
                    
                    print(f"\n{'='*60}")
                    print(f"🔥 CAL相似度空间操作")
                    print(f"  实验ID: {self.cal_config.get_experiment_id()}")
                    print(f"  负样本模式: {self.cal_config.negative_mode}")
                    print(f"  加权系数: alpha={self.cal_config.alpha}")
                    
                    # Q1: 生成负样本
                    negative_texts = self.negative_generator.generate(text)
                    print(f"  负样本: {negative_texts}")
                    
                    # 编码负样本
                    negative_features = encode_text_with_prompt_ensemble(
                        self.model, negative_texts, self.device
                    )
                    
                    # 应用CAL相似度空间操作
                    image_features_patches = image_features_all[:, 1:, :]
                    similarity_maps = self.cal_similarity_space.apply(
                        similarity_maps, image_features_patches, negative_features
                    )
                    
                    print(f"{'='*60}\n")
                
                # 生成热图
                from .clip import get_similarity_map
                heatmap_tensor = get_similarity_map(similarity_maps, (target_h, target_w))
                heatmap = heatmap_tensor[0, :, :, 0].detach().cpu().numpy()
                
                # 🔍 关键检查点：归一化后的heatmap
                print(f"  归一化后 heatmap: min={heatmap.min():.6f}, max={heatmap.max():.6f}, std={heatmap.std():.6f}")
                print(f"{'='*60}\n")
                
                print(f"🔍 热图值域: min={heatmap.min():.4f}, max={heatmap.max():.4f}, std={heatmap.std():.4f}")
                
                if np.isnan(heatmap).any() or np.isinf(heatmap).any():
                    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=1.0, neginf=0.0)
                
                heatmap = np.clip(heatmap, 0.0, 1.0)
                
                if return_features:
                    image_features_patches = image_features_all[:, 1:, :]
                    return heatmap, {
                        'image_features_all': image_features_all,
                        'image_features_patches': image_features_patches,
                        'text_features': text_features,
                        'similarity_maps': similarity_maps
                    }
                return heatmap

        except Exception as e:
            print(f"⚠️  热图生成失败: {e}")
            print("使用基础实现...")
            # 回退到全局特征
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 使用简单文本编码
            from .clip import tokenize
            if isinstance(text, str):
                text = [text]
            text_tokens = tokenize(text).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            if len(text) == 1:
                similarity_val = (image_features @ text_features.t()).item()
                # 创建均匀热图
                if isinstance(image_pil, Image.Image):
                    h, w = image_pil.size[1], image_pil.size[0]
                else:
                    h, w = 224, 224
                heatmap = np.full((h, w), similarity_val, dtype=np.float32)
            else:
                # 多文本：使用第一个文本
                similarity_val = (image_features @ text_features[0:1].t()).item()
                if isinstance(image_pil, Image.Image):
                    h, w = image_pil.size[1], image_pil.size[0]
                else:
                    h, w = 224, 224
                heatmap = np.full((h, w), similarity_val, dtype=np.float32)
            
            if return_features:
                return heatmap, {
                    'image_features': image_features,
                    'text_features': text_features
                }
            return heatmap

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  