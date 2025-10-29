# -*- coding: utf-8 -*-
"""
实验4损失函数
包含：分类、定位、稀疏性、正交性、对齐损失
"""

import torch
import torch.nn.functional as F


def classification_loss(Q, F_img, text_features, labels):
    """
    分类损失
    
    使用20个原子模式的加权池化进行分类
    
    Args:
        Q: [B, 196, 512, 20] 稀疏分解结果
        F_img: [B, 196, 512] 原始特征
        text_features: [N_classes, 512] 类别文本特征
        labels: [B] 标签
    
    Returns:
        loss: 标量
        logits: [B, N_classes] 预测logits
    """
    B, N, D, M = Q.shape
    
    # 确保所有输入具有相同的dtype
    target_dtype = F_img.dtype
    if Q.dtype != target_dtype:
        Q = Q.to(target_dtype)
    if text_features.dtype != target_dtype:
        text_features = text_features.to(target_dtype)
    
    # 聚合：用20个原子模式加权图像特征
    # 方法：每个模式对应一个注意力图
    attention = Q.abs().sum(dim=2)  # [B, 196, 20]
    attention = F.softmax(attention, dim=1)  # 在patch维度上归一化
    
    # 加权池化（每个原子模式独立池化）
    global_features = []
    for m in range(M):
        attn_m = attention[:, :, m].unsqueeze(-1)  # [B, 196, 1]
        feat_m = (F_img * attn_m).sum(dim=1)  # [B, 512]
        global_features.append(feat_m)
    
    # 聚合所有原子模式
    global_feat = torch.stack(global_features, dim=1).mean(dim=1)  # [B, 512]
    global_feat = F.normalize(global_feat, dim=1)
    
    # 与文本匹配
    text_features_norm = F.normalize(text_features, dim=1)
    logits = global_feat @ text_features_norm.T * 100  # [B, N_classes] 温度系数100
    
    # 交叉熵损失
    loss = F.cross_entropy(logits, labels)
    
    return loss, logits


def localization_loss(Q, F_img, bbox=None):
    """
    定位损失（如果有bbox标注）
    
    Args:
        Q: [B, 196, 512, 20]
        F_img: [B, 196, 512]
        bbox: [B, 4] (x1, y1, x2, y2) 归一化到[0,1]，可选
    
    Returns:
        loss: 标量
    """
    if bbox is None:
        # 没有bbox标注，返回0
        return torch.tensor(0.0, device=Q.device)
    
    B, N, D, M = Q.shape
    
    # 生成预测的attention map
    # 选择最discriminative的原子模式
    attention = Q.abs().sum(dim=2)  # [B, N, 20]
    attention_max = attention.max(dim=-1)[0]  # [B, N]
    
    # 动态计算grid大小并reshape到2D
    grid_size = int(N ** 0.5)
    assert grid_size * grid_size == N, f"Patches={N}不是完全平方数"
    pred_map = attention_max.reshape(B, grid_size, grid_size)  # [B, grid_size, grid_size]
    
    # 上采样到224x224
    pred_map = F.interpolate(
        pred_map.unsqueeze(1), 
        size=(224, 224), 
        mode='bilinear',
        align_corners=False
    ).squeeze(1)  # [B, 224, 224]
    
    # 归一化到[0, 1]
    pred_map = pred_map - pred_map.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    pred_map = pred_map / (pred_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-6)
    
    # 生成GT mask
    gt_mask = bbox_to_mask(bbox, size=(224, 224))  # [B, 224, 224]
    
    # IoU损失
    intersection = (pred_map * gt_mask).sum(dim=(1, 2))
    union = (pred_map + gt_mask - pred_map * gt_mask).sum(dim=(1, 2))
    iou = intersection / (union + 1e-6)
    
    loss = (1 - iou).mean()
    
    return loss


def sparsity_loss(Q):
    """
    稀疏性损失：鼓励大部分元素为0
    
    Args:
        Q: [B, 196, 512, 20]
    
    Returns:
        loss: 标量
    """
    # L1正则
    l1_loss = Q.abs().mean()
    
    # L0近似（可微分）- 非零元素的比例
    # 使用soft threshold
    threshold = 1e-3
    non_zero_ratio = (Q.abs() > threshold).float().mean()
    
    # 组合损失
    loss = l1_loss + 0.5 * non_zero_ratio
    
    return loss


def orthogonality_loss(Q):
    """
    正交性损失：20个原子模式应该独立
    
    Args:
        Q: [B, 196, 512, 20]
    
    Returns:
        loss: 标量
    """
    B, N, D, M = Q.shape
    
    # 在batch和patch维度上平均，得到每个模式的"代表向量"
    Q_avg = Q.mean(dim=(0, 1))  # [512, 20]
    
    # 归一化
    Q_avg = F.normalize(Q_avg, dim=0, eps=1e-6)
    
    # 计算Gram矩阵
    gram = Q_avg.T @ Q_avg  # [20, 20]
    
    # 应该接近单位矩阵（对角线为1，非对角线为0）
    identity = torch.eye(M, device=Q.device)
    
    # MSE损失
    loss = F.mse_loss(gram, identity)
    
    return loss


def alignment_loss(Q_text, Q_img, F_img):
    """
    对齐损失：text-guided和image-only两条路径的attention应该相似
    
    这使得image-only分解器可以学习text-guided的知识，从而泛化到unseen类
    
    Args:
        Q_text: [B, 196, 512, 20] text引导的分解
        Q_img: [B, 196, 512, 20] 纯图像的分解
        F_img: [B, 196, 512] 原始特征
    
    Returns:
        loss: 标量
    """
    # 生成attention maps
    attn_text = (Q_text * F_img.unsqueeze(-1)).sum(dim=2).abs()  # [B, 196, 20]
    attn_img = (Q_img * F_img.unsqueeze(-1)).sum(dim=2).abs()  # [B, 196, 20]
    
    # 归一化
    attn_text = F.normalize(attn_text, dim=1, eps=1e-6)  # 在patch维度上归一化
    attn_img = F.normalize(attn_img, dim=1, eps=1e-6)
    
    # 计算相似度矩阵（允许模式之间的软匹配）
    # [B, 20, 196] @ [B, 196, 20] → [B, 20, 20]
    sim_matrix = torch.bmm(
        attn_text.transpose(1, 2),  # [B, 20, 196]
        attn_img  # [B, 196, 20]
    )  # [B, 20, 20]
    
    # 最大化最优匹配（每个text模式找到最相似的img模式）
    max_sim_per_text = sim_matrix.max(dim=-1)[0]  # [B, 20]
    max_sim_per_img = sim_matrix.max(dim=-2)[0]  # [B, 20]
    
    # 双向对齐
    avg_max_sim = (max_sim_per_text.mean() + max_sim_per_img.mean()) / 2
    
    loss = 1 - avg_max_sim
    
    return loss


def bbox_to_mask(bbox, size=(224, 224)):
    """
    将bbox转换为mask
    
    Args:
        bbox: [B, 4] (x1, y1, x2, y2) in [0, 1]
        size: (H, W)
    
    Returns:
        mask: [B, H, W]
    """
    B = len(bbox)
    H, W = size
    mask = torch.zeros(B, H, W, device=bbox.device)
    
    for b in range(B):
        x1 = int(bbox[b, 0].item() * W)
        y1 = int(bbox[b, 1].item() * H)
        x2 = int(bbox[b, 2].item() * W)
        y2 = int(bbox[b, 3].item() * H)
        
        # 确保坐标有效
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H))
        
        if x2 > x1 and y2 > y1:
            mask[b, y1:y2, x1:x2] = 1.0
    
    return mask


def compute_total_loss(Q_text, Q_img, F_img, text_features, labels, bbox, config):
    """
    计算总损失
    
    Args:
        Q_text: [B, 196, 512, 20] text引导的分解
        Q_img: [B, 196, 512, 20] 图像分解
        F_img: [B, 196, 512] 原始特征
        text_features: [N_classes, 512] 类别文本特征
        labels: [B] 标签
        bbox: [B, 4] or None
        config: 配置对象
    
    Returns:
        total_loss: 标量
        loss_dict: 各项损失的字典
    """
    # 1. 分类损失（text-guided路径）
    loss_cls_text, logits_text = classification_loss(
        Q_text, F_img, text_features, labels
    )
    
    # 2. 分类损失（image-only路径，辅助）
    loss_cls_img, logits_img = classification_loss(
        Q_img, F_img, text_features, labels
    )
    
    # 3. 定位损失
    loss_loc_text = localization_loss(Q_text, F_img, bbox)
    loss_loc_img = localization_loss(Q_img, F_img, bbox)
    loss_loc = loss_loc_text + 0.5 * loss_loc_img
    
    # 4. 稀疏性损失
    loss_sparse_text = sparsity_loss(Q_text)
    loss_sparse_img = sparsity_loss(Q_img)
    loss_sparse = loss_sparse_text + loss_sparse_img
    
    # 5. 正交性损失
    loss_ortho_text = orthogonality_loss(Q_text)
    loss_ortho_img = orthogonality_loss(Q_img)
    loss_ortho = loss_ortho_text + loss_ortho_img
    
    # 6. 对齐损失
    loss_align = alignment_loss(Q_text, Q_img, F_img)
    
    # 总损失
    total_loss = (
        config.w_cls * loss_cls_text +
        config.w_cls * 0.5 * loss_cls_img +
        config.w_loc * loss_loc +
        config.w_sparse * loss_sparse +
        config.w_ortho * loss_ortho +
        config.w_align * loss_align
    )
    
    # 损失字典
    loss_dict = {
        'total': total_loss.item(),
        'cls_text': loss_cls_text.item(),
        'cls_img': loss_cls_img.item(),
        'loc': loss_loc.item() if isinstance(loss_loc, torch.Tensor) else loss_loc,
        'sparse': loss_sparse.item(),
        'ortho': loss_ortho.item(),
        'align': loss_align.item(),
        'acc_text': (logits_text.argmax(1) == labels).float().mean().item(),
        'acc_img': (logits_img.argmax(1) == labels).float().mean().item()
    }
    
    return total_loss, loss_dict


def test_losses():
    """测试损失函数"""
    print("测试损失函数...")
    
    B, N, D, M = 4, 196, 512, 20
    num_classes = 10
    
    # 模拟数据
    Q_text = torch.randn(B, N, D, M)
    Q_img = torch.randn(B, N, D, M)
    F_img = torch.randn(B, N, D)
    text_features = torch.randn(num_classes, D)
    labels = torch.randint(0, num_classes, (B,))
    bbox = torch.rand(B, 4)
    
    # 测试各项损失
    loss_cls, logits = classification_loss(Q_text, F_img, text_features, labels)
    print(f"分类损失: {loss_cls.item():.4f}, Logits形状: {logits.shape}")
    
    loss_loc = localization_loss(Q_text, F_img, bbox)
    print(f"定位损失: {loss_loc.item():.4f}")
    
    loss_sp = sparsity_loss(Q_text)
    print(f"稀疏性损失: {loss_sp.item():.4f}")
    
    loss_orth = orthogonality_loss(Q_text)
    print(f"正交性损失: {loss_orth.item():.4f}")
    
    loss_ali = alignment_loss(Q_text, Q_img, F_img)
    print(f"对齐损失: {loss_ali.item():.4f}")
    
    print("测试通过！")


if __name__ == "__main__":
    test_losses()

