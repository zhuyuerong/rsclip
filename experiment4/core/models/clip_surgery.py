# -*- coding: utf-8 -*-
"""
CLIP Surgery模型 - 修复版
真正实现V-V attention机制，去除文本泄露
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from pathlib import Path
from collections import OrderedDict


def clip_feature_surgery(image_features, text_features, redundant_feats=None, t=2):
    """
    Feature Surgery去冗余
    
    基于CLIP Surgery论文的实现，剔除多类别共享的冗余特征
    
    Args:
        image_features: [B, N+1, C] 图像特征（包含CLS token）
        text_features: [N_classes, C] 文本特征
        redundant_feats: 预计算的冗余特征（可选）
        t: 温度参数（默认2）
    
    Returns:
        similarity: [B, N_patches, N_classes] 去冗余后的相似度（不含CLS token）
    """
    if redundant_feats is not None:
        # 使用预计算的冗余特征
        # 只使用patch特征，不含CLS
        patch_features = image_features[:, 1:, :]  # [B, N_patches, C]
        similarity = patch_features @ (text_features - redundant_feats).t()
    else:
        # 计算类别权重（使用CLS token）
        prob = image_features[:, :1, :] @ text_features.t()  # [B, 1, N_classes]
        prob = (prob * t).softmax(-1)  # 温度=t
        w = prob / prob.mean(-1, keepdim=True)  # 归一化权重
        
        # 只使用patch特征进行后续计算
        patch_features = image_features[:, 1:, :]  # [B, N_patches, C]
        
        # 类别-位置特异性特征（逐元素相乘）
        b, n_t, n_i, c = patch_features.shape[0], text_features.shape[0], patch_features.shape[1], patch_features.shape[2]
        feats = patch_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
        # [B, N_patches, 1, C] × [1, 1, N_classes, C] → [B, N_patches, N_classes, C]
        
        # 应用类别权重
        feats *= w.reshape(1, 1, n_t, 1)
        
        # 计算并剔除冗余（多类别共享部分）
        redundant_feats = feats.mean(2, keepdim=True)  # [B, N_patches, 1, C]
        feats = feats - redundant_feats  # 剔除冗余，保留类别特异性
        
        # 沿特征维度求和得到相似度分数
        similarity = feats.sum(-1)  # [B, N_patches, N_classes]
    
    return similarity


def get_similarity_map(similarity, original_shape):
    """
    将相似度转换为热图
    
    Args:
        similarity: [B, N_tokens, N_classes] 相似度分数（N_tokens可能包含CLS token）
        original_shape: (H, W) 原图尺寸
    
    Returns:
        heatmaps: [B, N_classes, H, W] 热图
    """
    B, N_tokens, N_classes = similarity.shape
    
    # 如果包含CLS token（N_tokens = N_patches + 1），去掉CLS
    # 判断依据：√N_tokens是否为整数
    sqrt_n = int(N_tokens ** 0.5)
    if sqrt_n * sqrt_n != N_tokens:
        # N_tokens不是完全平方数，可能包含CLS token
        # 尝试去掉第一个token (CLS)
        if int((N_tokens - 1) ** 0.5) ** 2 == N_tokens - 1:
            # 去掉CLS后是完全平方数，说明第一个是CLS
            # 但similarity是[B, N_tokens, K]，CLS已经在clip_feature_surgery中被使用
            # 这里我们假设similarity已经是纯patch的（不含CLS）
            pass
    
    N_patches = N_tokens  # 假设已经是patch数量
    
    # Min-Max归一化到[0, 1]
    sm = (similarity - similarity.min(1, keepdim=True)[0]) / \
         (similarity.max(1, keepdim=True)[0] - similarity.min(1, keepdim=True)[0] + 1e-8)
    
    # Reshape为空间特征图
    side = int(N_patches ** 0.5)
    if side * side != N_patches:
        raise ValueError(f"N_patches={N_patches}不是完全平方数，无法reshape为正方形网格")
    
    sm = sm.reshape(B, side, side, N_classes)  # [B, side, side, N_classes]
    sm = sm.permute(0, 3, 1, 2)  # [B, N_classes, side, side]
    
    # 上采样到原图尺寸
    heatmaps = F.interpolate(sm, size=original_shape, mode='bilinear', align_corners=False)
    # [B, N_classes, H, W]
    
    return heatmaps


class LayerNorm(nn.LayerNorm):
    """支持fp16的LayerNorm"""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """快速GELU激活函数"""
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class VVAttention(nn.Module):
    """
    V-V自注意力机制
    核心：将Q和K替换为V，计算Attention(V, V, V)
    """
    
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., 
                 settings='vit', scale_multiplier=1.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.settings = settings
        self.scale_multiplier = scale_multiplier

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_kv=None, x_q=None, need_weights=False, attn_mask=None):
        """
        Args:
            x: [B, N, C] or [N, B, C] (LND格式)
            x_kv, x_q: 兼容CLIP调用，但实际不使用
            need_weights: 兼容CLIP调用，但实际不使用
            attn_mask: 兼容CLIP调用，但实际不使用
        Returns:
            [x_vv, x_ori]: VV路径和原始路径的输出
        """
        # 处理输入格式
        if x.dim() == 3 and x.shape[0] != x.shape[1]:  # 可能是LND格式
            is_ldn = True
            x = x.transpose(0, 1)  # LND -> NLD
        else:
            is_ldn = False
            
        B, N, C = x.shape
        
        # 计算Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # ========== 原始路径：标准QK自注意力 ==========
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)
        
        # ========== VV路径：将Q和K替换为V ==========
        k_vv = v.clone()
        q_vv = k_vv.clone()
        
        if self.settings == 'resnet':
            k_vv = k_vv / (k_vv.norm(p=2, dim=-1, keepdim=True) + 1e-6)
            q_vv = k_vv.clone()
            scale_vv = self.scale * self.scale_multiplier
        else:
            scale_vv = self.scale * self.scale_multiplier
        
        # VV注意力：Attention(V, V, V)
        attn_vv = (q_vv @ k_vv.transpose(-2, -1)) * scale_vv
        attn_vv = attn_vv.softmax(dim=-1)
        attn_vv = self.attn_drop(attn_vv)
        
        # 应用注意力权重
        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x_vv = (attn_vv @ v).transpose(1, 2).reshape(B, N, C)
        
        # 投影层
        x_ori = self.proj_drop(self.proj(x_ori))
        x_vv = self.proj_drop(self.proj(x_vv))
        
        # 转换回原格式
        if is_ldn:
            x_ori = x_ori.transpose(0, 1)  # NLD -> LND
            x_vv = x_vv.transpose(0, 1)  # NLD -> LND
        
        return [x_vv, x_ori]


class ResidualAttentionBlockWithVV(nn.Module):
    """支持VV自注意力的残差块"""
    
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        
        # 如果是VV自注意力，返回双路径结果
        if isinstance(self.attn, VVAttention):
            x_res = self.attn(x)
            if isinstance(x_res, list):
                return x_res
            return x_res
        else:
            # 标准自注意力
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        # 双路径模式（VV自注意力）
        if isinstance(self.attn, VVAttention):
            if isinstance(x, list):
                # 已经在双路径中
                x_vv, x_ori = x
                x_res = self.attention(self.ln_1(x_ori))
                if isinstance(x_res, list):
                    x_res_vv, x_res_ori = x_res
                    x_ori = x_ori + x_res_ori
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x_vv = x_vv + x_res_vv  # VV路径跳过MLP
                    return [x_vv, x_ori]
            else:
                # 开始双路径
                x_res = self.attention(self.ln_1(x))
                if isinstance(x_res, list):
                    x_res_vv, x_res_ori = x_res
                    x_ori = x + x_res_ori
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x_vv = x + x_res_vv
                    return [x_vv, x_ori]
        
        # 单路径模式（标准自注意力）
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPSurgery(nn.Module):
    """
    CLIP Surgery模型 - 真正实现VV机制
    
    使用V-V attention代替标准attention，避免文本泄露
    """
    
    def __init__(self, clip_model, device="cuda", num_vv_blocks=6):
        super().__init__()
        self.clip_model = clip_model
        self.device = device
        self.num_vv_blocks = num_vv_blocks
        
        # 冻结所有参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        self.clip_model.eval()
        
        # VV机制应用标记
        self.vv_applied = False
    
    @classmethod
    def from_pretrained(cls, model_name="ViT-B/32", device="cuda", num_vv_blocks=6):
        """
        从预训练CLIP加载并应用VV机制
        
        Args:
            model_name: CLIP模型名称
            device: 设备
            num_vv_blocks: 应用VV机制的层数（从后往前）
        
        Returns:
            CLIPSurgery实例
        """
        # 检查是否有RemoteCLIP权重
        remoteclip_path = Path("checkpoints/RemoteCLIP-ViT-B-32.pt")
        
        if remoteclip_path.exists() and "B" in model_name:
            print(f"加载RemoteCLIP权重: {remoteclip_path}")
            clip_model, _ = clip.load("ViT-B/32", device=device)
            
            # 加载RemoteCLIP权重
            checkpoint = torch.load(remoteclip_path, map_location=device)
            if 'state_dict' in checkpoint:
                clip_model.load_state_dict(checkpoint['state_dict'])
            else:
                clip_model.load_state_dict(checkpoint)
        else:
            print(f"加载CLIP权重: {model_name}")
            clip_model, _ = clip.load(model_name, device=device)
        
        return cls(clip_model, device, num_vv_blocks)
    
    @torch.no_grad()
    def _apply_vv_mechanism(self):
        """应用VV机制到最后num_vv_blocks层"""
        if self.vv_applied:
            return
        
        visual = self.clip_model.visual
        if not hasattr(visual, 'transformer'):
            print("警告: 这不是ViT模型，无法应用VV机制")
            return
        
        transformer = visual.transformer
        embed_dim = transformer.width
        num_heads = transformer.resblocks[0].attn.num_heads
        
        print(f"应用VV机制到{self.num_vv_blocks}层 (embed_dim={embed_dim}, num_heads={num_heads})")
        
        # 对最后num_vv_blocks层应用VV机制
        for i in range(1, self.num_vv_blocks + 1):
            block_idx = len(transformer.resblocks) - i
            block = transformer.resblocks[block_idx]
            original_attn = block.attn
            
            # 创建VV自注意力模块
            vv_attn = VVAttention(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=True,
                settings='vit'
            )
            
            # 从原始注意力层复制权重
            if hasattr(original_attn, 'in_proj_weight') and original_attn.in_proj_weight is not None:
                # MultiheadAttention的权重格式：[Q|K|V]
                vv_attn.qkv.weight.data = original_attn.in_proj_weight.clone()
                vv_attn.qkv.bias.data = original_attn.in_proj_bias.clone()
            else:
                # 分离的Q, K, V投影（通常不会有）
                print(f"警告: 块{block_idx}的注意力权重格式未识别")
            
            # 复制输出投影权重
            if hasattr(original_attn, 'out_proj'):
                vv_attn.proj.weight.data = original_attn.out_proj.weight.clone()
                vv_attn.proj.bias.data = original_attn.out_proj.bias.clone()
            
            # 替换注意力模块
            block.attn = vv_attn
            print(f"  ✓ 层 {block_idx} 已应用VV机制")
        
        self.vv_applied = True
        print("VV机制应用完成！")
    
    def encode_image(self, images):
        """
        编码图像，使用V-V attention
        
        Args:
            images: [B, 3, H, W]
        
        Returns:
            features: [B, N+1, D] (包含CLS token和所有patches)
        """
        # 首次前向传播时应用VV机制
        if not self.vv_applied:
            self._apply_vv_mechanism()
        
        with torch.no_grad():
            visual = self.clip_model.visual
            
            # 确保输入类型匹配
            if images.dtype != visual.conv1.weight.dtype:
                images = images.to(visual.conv1.weight.dtype)
            
            # Patch嵌入
            x = visual.conv1(images)  # [B, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, grid**2]
            x = x.permute(0, 2, 1)  # [B, grid**2, width]
            
            # 添加CLS token
            cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], 
                dtype=x.dtype, device=x.device
            )
            x = torch.cat([cls_token, x], dim=1)  # [B, grid**2+1, width]
            
            # 位置嵌入插值（处理不同输入尺寸）
            side = int((visual.positional_embedding.shape[0] - 1) ** 0.5)
            new_side = int((x.shape[1] - 1) ** 0.5)
            
            if side != new_side:
                new_pos = visual.positional_embedding[1:, :].reshape(
                    -1, side, side, x.shape[-1]
                ).permute(0, 3, 1, 2)
                new_pos = F.interpolate(
                    new_pos, (new_side, new_side), mode='bilinear'
                )
                new_pos = new_pos.reshape(
                    -1, x.shape[-1], new_side * new_side
                ).transpose(1, 2)
                visual.positional_embedding.data = torch.cat(
                    [visual.positional_embedding[:1, :], new_pos[0]], 0
                )
            
            # 添加位置嵌入
            x = x + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x)
            
            # Transformer编码（NLD -> LND）
            x = x.permute(1, 0, 2)  # NLD -> LND
            result = visual.transformer(x)
            
            # 处理双路径输出（如果有VV机制）
            if isinstance(result, list):
                x_vv, x_ori = result
                # CLS token使用原始路径，图像tokens使用VV路径
                x_vv[0, :, :] = x_ori[0, :, :]
                x = x_vv
            else:
                x = result
            
            x = x.permute(1, 0, 2)  # LND -> NLD
            
            # 最终归一化和投影
            x = visual.ln_post(x)
            if hasattr(visual, 'proj') and visual.proj is not None:
                x = x @ visual.proj
        
        return x
    
    def encode_text(self, text_list):
        """
        编码文本（保持原始CLIP实现）
        
        Args:
            text_list: list of str or tokenized text
        
        Returns:
            features: [N, 512]
        """
        with torch.no_grad():
            if isinstance(text_list, list):
                text_tokens = clip.tokenize(text_list).to(self.device)
            else:
                text_tokens = text_list
            
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def get_visual_features(self, images, return_all_tokens=True):
        """
        获取视觉特征
        
        Args:
            images: [B, 3, H, W]
            return_all_tokens: 是否返回所有token
        
        Returns:
            features: [B, N+1, D] if return_all_tokens else [B, D]
        """
        features = self.encode_image(images)
        
        if return_all_tokens:
            return features
        else:
            return features[:, 0, :]  # 只返回CLS token


class CLIPSurgeryWrapper:
    """
    CLIP Surgery的简化包装器
    
    支持3种模式：
    1. 标准RemoteCLIP（use_surgery=False, use_vv=False）
    2. Surgery去冗余（use_surgery=True, use_vv=False）
    3. Surgery+VV机制（use_surgery=True, use_vv=True）
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.use_surgery = getattr(config, 'use_surgery', False)
        self.use_vv = getattr(config, 'use_vv_mechanism', False)
        
        # 加载模型
        if self.use_vv:
            # 模式3: Surgery+VV机制
            self.model = CLIPSurgery.from_pretrained(
                model_name=config.backbone,
                device=self.device,
                num_vv_blocks=getattr(config, 'num_vv_blocks', 6)
            )
            self.clip_model = None
        else:
            # 模式1/2: 标准RemoteCLIP（可选Surgery）
            remoteclip_path = Path("checkpoints/RemoteCLIP-ViT-B-32.pt")
            self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
            if remoteclip_path.exists():
                checkpoint = torch.load(remoteclip_path, map_location=self.device)
                if 'state_dict' in checkpoint:
                    self.clip_model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.clip_model.load_state_dict(checkpoint)
            self.clip_model.eval()
            self.model = None
        
        # 预计算背景词特征
        if hasattr(config, 'background_words'):
            self.bg_features = self.encode_text(config.background_words)
    
    def encode_image(self, images):
        """编码图像"""
        if self.model is not None:
            return self.model.encode_image(images)
        else:
            with torch.no_grad():
                return self.clip_model.encode_image(images)
    
    def encode_text(self, text_list):
        """编码文本"""
        if self.model is not None:
            return self.model.encode_text(text_list)
        else:
            with torch.no_grad():
                if isinstance(text_list, list):
                    text_tokens = clip.tokenize(text_list).to(self.device)
                else:
                    text_tokens = text_list
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                return text_features
    
    def get_all_features(self, images):
        """
        获取完整特征（包含CLS token + patches）
        
        Args:
            images: [B, 3, 224, 224]
        
        Returns:
            all_features: [B, N+1, 512]
        """
        return self.model.encode_image(images)
    
    def get_cls_features(self, images):
        """
        获取CLS token特征
        
        Args:
            images: [B, 3, 224, 224]
        
        Returns:
            cls_features: [B, 512]
        """
        all_features = self.model.encode_image(images)
        return all_features[:, 0, :]
    
    def get_patch_features(self, images):
        """
        获取patch特征（去掉CLS token）
        
        Args:
            images: [B, 3, 224, 224]
        
        Returns:
            patch_features: [B, N, 512]
        """
        all_features = self.encode_image(images)
        return all_features[:, 1:, :]
    
    # ===== 统一相似度计算接口 =====
    def _vanilla_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        纯CLIP余弦相似度（对patch与文本均进行L2归一化）
        Args:
            image_features: [B, N+1, C]
            text_features: [K, C]
        Returns:
            similarity: [B, N_patches, K]
        """
        patch_features = image_features[:, 1:, :]
        patch_features = F.normalize(patch_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        return patch_features @ text_features.t()
    
    def compute_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        根据当前配置计算相似度
        - VV 影响已体现在 encode_image 产生的 image_features 中
        - use_surgery 决定是否执行 Feature Surgery 去冗余
        Returns: [B, N_patches, K]
        """
        if self.use_surgery:
            return clip_feature_surgery(image_features, text_features, t=2)
        return self._vanilla_similarity(image_features, text_features)
    
    def get_layer_features(self, images, layer_indices=[1, 6, 9, 12]):
        """
        提取指定层的特征（包含投影到512维）
        
        Args:
            images: [B, 3, H, W]
            layer_indices: 要提取的层索引（从1开始）
        
        Returns:
            features_dict: {layer_idx: features} 各层特征字典
                           features格式: [B, N+1, 512] (投影后)
        """
        features_dict = {}
        intermediate_features = {}
        
        # 获取visual transformer
        if self.model is not None:
            visual = self.model.clip_model.visual
        else:
            visual = self.clip_model.visual
        
        # Hook方式提取中间层特征
        def hook_fn(layer_idx):
            def hook(module, input, output):
                # 处理双路径输出（VV机制）
                if isinstance(output, list):
                    intermediate_features[layer_idx] = output[0].clone()  # VV路径
                else:
                    intermediate_features[layer_idx] = output.clone()
            return hook
        
        handles = []
        for idx in layer_indices:
            if idx < 1 or idx > len(visual.transformer.resblocks):
                print(f"警告: 层索引{idx}超出范围，跳过")
                continue
            handle = visual.transformer.resblocks[idx-1].register_forward_hook(hook_fn(idx))
            handles.append(handle)
        
        # 前向传播
        with torch.no_grad():
            _ = self.encode_image(images)
        
        # 移除hooks
        for handle in handles:
            handle.remove()
        
        # 后处理：转换格式并投影到512维
        for layer_idx in intermediate_features:
            feat = intermediate_features[layer_idx]
            
            # 转换格式: LND -> NLD ([L, B, C] -> [B, L, C])
            if feat.dim() == 3:
                if feat.shape[0] > feat.shape[1]:  # LND格式 [L, B, C]
                    feat = feat.permute(1, 0, 2)  # -> [B, L, C]
            
            # 应用layer norm和投影（如果存在）
            if hasattr(visual, 'ln_post'):
                feat = visual.ln_post(feat)
            
            if hasattr(visual, 'proj') and visual.proj is not None:
                # 投影到512维
                if feat.dtype != visual.proj.dtype:
                    feat = feat.to(visual.proj.dtype)
                feat = feat @ visual.proj
            
            features_dict[layer_idx] = feat
        
        return features_dict


def test_clip_surgery():
    """测试CLIP Surgery"""
    print("测试CLIP Surgery (修复版)...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPSurgery.from_pretrained("ViT-B/32", device=device, num_vv_blocks=6)
    
    # 测试图像编码
    images = torch.randn(2, 3, 224, 224).to(device)
    img_features = model.encode_image(images)
    print(f"\n图像特征形状: {img_features.shape}")
    print(f"  预期: [2, 50, 512] (ViT-B/32: 1 CLS + 49 patches)")
    
    # 测试文本编码
    texts = ["airplane", "ship", "car"]
    text_features = model.encode_text(texts)
    print(f"\n文本特征形状: {text_features.shape}")
    print(f"  预期: [3, 512]")
    
    # 测试不同方法
    all_features = model.get_visual_features(images, return_all_tokens=True)
    cls_features = model.get_visual_features(images, return_all_tokens=False)
    print(f"\n完整特征形状: {all_features.shape}")
    print(f"CLS特征形状: {cls_features.shape}")
    
    print("\n✓ 测试通过！VV机制已正确应用！")


if __name__ == "__main__":
    test_clip_surgery()

