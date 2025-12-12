# src/experiments/exp2/surgeryclip_backbone.py
import os
import sys
from typing import Dict, List

import torch
import torch.nn as nn

# ====== 保证能找到你的 SurgeryCLIP build_model ======
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 这条路径根据你项目结构适当调整
from src.competitors.clip_methods.surgeryclip.build_model import build_model as build_surgeryclip

# GroundingDINO 里用的 NestedTensor 工具
gdino_root = os.path.join(project_root, "src", "experiments", "exp2", "Open-GroundingDino-main")
if gdino_root not in sys.path:
    sys.path.insert(0, gdino_root)

from groundingdino.util.misc import NestedTensor
# PositionEmbeddingSine 在 backbone/position_encoding.py 中
# 直接导入，因为已经在 sys.path 中添加了 gdino_root
try:
    from models.GroundingDINO.backbone.position_encoding import PositionEmbeddingSine
except ImportError:
    # 如果上面的导入失败，尝试使用 importlib
    import importlib.util
    position_encoding_path = os.path.join(gdino_root, "models", "GroundingDINO", "backbone", "position_encoding.py")
    spec = importlib.util.spec_from_file_location("position_encoding", position_encoding_path)
    position_encoding_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(position_encoding_module)
    PositionEmbeddingSine = position_encoding_module.PositionEmbeddingSine


class SurgeryCLIPBackbone(nn.Module):
    """
    用 SurgeryCLIP 的 visual encoder 作为 GroundingDINO 的 backbone。
    最小版本：只输出最后一层 patch feature，单尺度。
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda", train_backbone: bool = False):
        super().__init__()
        self.device = device

        # ====== 1. 加载 SurgeryCLIP 模型 ======
        # build_model 返回 (model, preprocess)
        self.clip_model, self.preprocess = build_surgeryclip(
            model_name="surgeryclip",
            checkpoint_path=checkpoint_path,
            device=device,
        )

        # 只用视觉部分
        self.visual = self.clip_model.visual

        # 是否冻结 backbone
        if not train_backbone:
            for p in self.visual.parameters():
                p.requires_grad_(False)

        # ====== 2. 准备一个位置编码（和 GDINO 一致的 Sine PE） ======
        # 这里的 num_pos_feats = embed_dim // 2
        embed_dim = self.visual.embed_dim  # 对 ViT 来说是 transformer width
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=embed_dim // 2,
            normalize=True,
        )

        self.num_channels = embed_dim  # 给 transformer encoder 用

    @torch.no_grad()
    def _encode_image_tokens(self, images: torch.Tensor):
        """
        使用 SurgeryCLIP 的 encode_image_with_all_tokens，得到 cls + patch tokens

        Args:
            images: [B, 3, H, W] 已经做过预处理的图像张量（和 CLIP 一致）

        Returns:
            patch_features: [B, D, H_patch, W_patch]
        """
        # encode_image_with_all_tokens: [B, 1+N², D]
        if hasattr(self.visual, "encode_image_with_all_tokens"):
            tokens = self.visual.encode_image_with_all_tokens(images.to(self.device))
        else:
            # fallback：用 encode_image，然后自己 reshape（不推荐，优先改好 visual）
            raise NotImplementedError("visual.encode_image_with_all_tokens 缺失")

        B, L, D = tokens.shape
        N_sq = L - 1
        N = int(N_sq ** 0.5)
        assert N * N == N_sq, f"patch 数量不是平方数: {N_sq}"

        # 去掉 CLS，reshape 成 feature map
        patch_tokens = tokens[:, 1:, :]                   # [B, N², D]
        patch_tokens = patch_tokens.reshape(B, N, N, D)   # [B, N, N, D]
        patch_features = patch_tokens.permute(0, 3, 1, 2) # [B, D, N, N]

        return patch_features

    def forward(self, samples: NestedTensor) -> Dict[str, NestedTensor]:
        """
        GroundingDINO backbone 接口：
        输入 NestedTensor(images, mask)
        输出 一个 dict: {level_name: NestedTensor(feat, mask)}

        为简单起见，我们只返回一个 level："0"
        """
        x = samples.tensors  # [B, 3, H, W]
        mask = samples.mask  # [B, H, W] or None

        # 注意：CLIP 预处理通常是 Resize + CenterCrop + Normalize
        # 这里最简单的做法是：假设传进来的已经和 CLIP 训练分布接近（后续可以精细处理）
        patch_features = self._encode_image_tokens(x)  # [B, D, N, N]

        # 创建对应大小的 mask（如果原 mask 存在，需要插值）
        if mask is not None:
            # 将 mask 从 HxW 缩放到 NxN
            mask = torch.nn.functional.interpolate(
                mask[None].float(), size=patch_features.shape[-2:], mode="nearest"
            )[0].bool()
        else:
            # 如果没有 mask，创建一个全 False 的 mask
            B, D, N, N = patch_features.shape
            mask = torch.zeros(B, N, N, dtype=torch.bool, device=patch_features.device)

        nested = NestedTensor(patch_features, mask)

        return {"0": nested}  # 单尺度

    def get_num_channels(self):
        return self.num_channels

    def get_position_embedding(self, nested: NestedTensor):
        """
        供上层 Joiner 调用：根据 NestedTensor 生成位置编码
        """
        return self.position_embedding(nested)


def build_surgeryclip_backbone(args):
    """
    给 groundingdino.models.build_model 调用的入口。
    返回：
      - backbone: SurgeryCLIPBackbone 实例
      - num_channels: 通道数（列表形式，因为 GroundingDINO 期望列表）
    """
    checkpoint_path = getattr(args, "surgeryclip_ckpt", None)
    if checkpoint_path is None:
        raise ValueError("请在 config 里设置 args.surgeryclip_ckpt 为 SurgeryCLIP 权重路径")

    backbone = SurgeryCLIPBackbone(
        checkpoint_path=checkpoint_path,
        device=args.device,
        train_backbone=getattr(args, "train_surgeryclip_backbone", False),
    )

    # GroundingDINO 期望 num_channels 是列表
    # 对于单尺度，返回一个元素的列表
    num_channels = [backbone.num_channels]
    
    return backbone, num_channels

