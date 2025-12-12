# experiments/exp2/run_gdino_sanity.py
import os
import sys
from PIL import Image
import torch
import torchvision.transforms as T

# 设置使用镜像源（如果网络有问题）
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = '1'

# ====== 1. 把 exp2 加到 sys.path ======
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
exp2_root = os.path.join(project_root, "experiments", "exp2")
open_gdino_root = os.path.join(exp2_root, "Open-GroundingDino-main")

# 添加 Open-GroundingDino-main 到路径，这样可以从 groundingdino 导入
if open_gdino_root not in sys.path:
    sys.path.insert(0, open_gdino_root)

# ====== 2. 导入 GroundingDINO 的构建函数 & 配置 ======
from groundingdino.util.slconfig import SLConfig
from groundingdino.models import build_model
from groundingdino.util.utils import clean_state_dict

def is_checkpoint_valid(ckpt_path):
    """验证 checkpoint 文件是否完整"""
    try:
        import torch
        torch.load(ckpt_path, map_location="cpu")
        return True
    except:
        return False

def load_model(config_path, checkpoint_path, device="cuda"):
    # 解析配置
    args = SLConfig.fromfile(config_path)
    args.device = device
    
    # 设置默认值（如果配置文件中没有定义）
    # 模型相关
    if not hasattr(args, 'aux_loss'):
        args.aux_loss = False
    if not hasattr(args, 'iter_update'):
        args.iter_update = False
    
    # Matcher 相关
    if not hasattr(args, 'matcher_type'):
        args.matcher_type = 'HungarianMatcher'
    if not hasattr(args, 'set_cost_class'):
        args.set_cost_class = 2.0
    if not hasattr(args, 'set_cost_bbox'):
        args.set_cost_bbox = 5.0
    if not hasattr(args, 'set_cost_giou'):
        args.set_cost_giou = 2.0
    if not hasattr(args, 'focal_alpha'):
        args.focal_alpha = 0.25
    
    # Loss 相关
    if not hasattr(args, 'cls_loss_coef'):
        args.cls_loss_coef = 2.0
    if not hasattr(args, 'bbox_loss_coef'):
        args.bbox_loss_coef = 5.0
    if not hasattr(args, 'giou_loss_coef'):
        args.giou_loss_coef = 2.0
    if not hasattr(args, 'enc_loss_coef'):
        args.enc_loss_coef = 1.0
    if not hasattr(args, 'interm_loss_coef'):
        args.interm_loss_coef = 1.0
    if not hasattr(args, 'mask_loss_coef'):
        args.mask_loss_coef = 1.0
    if not hasattr(args, 'dice_loss_coef'):
        args.dice_loss_coef = 1.0
    
    # Focal loss 相关
    if not hasattr(args, 'focal_alpha'):
        args.focal_alpha = 0.25
    if not hasattr(args, 'focal_gamma'):
        args.focal_gamma = 2.0
    
    # Decoder 相关
    if not hasattr(args, 'decoder_sa_type'):
        args.decoder_sa_type = 'sa'
    if not hasattr(args, 'decoder_module_seq'):
        args.decoder_module_seq = ['sa', 'ca', 'ffn']
    elif isinstance(args.decoder_module_seq, str):
        # 如果是字符串，转换为列表
        args.decoder_module_seq = [args.decoder_module_seq]
    if not hasattr(args, 'dec_pred_class_embed_share'):
        args.dec_pred_class_embed_share = True
    if not hasattr(args, 'use_detached_boxes_dec_out'):
        args.use_detached_boxes_dec_out = False
    
    # 其他
    if not hasattr(args, 'nms_iou_threshold'):
        args.nms_iou_threshold = -1
    if not hasattr(args, 'match_unstable_error'):
        args.match_unstable_error = True
    if not hasattr(args, 'dn_scalar'):
        args.dn_scalar = 100
    if not hasattr(args, 'no_interm_box_loss'):
        args.no_interm_box_loss = False
    if not hasattr(args, 'masks'):
        args.masks = False
    
    # PostProcess 相关
    if not hasattr(args, 'num_select'):
        args.num_select = 300
    if not hasattr(args, 'nms_iou_threshold'):
        args.nms_iou_threshold = -1
    if not hasattr(args, 'use_coco_eval'):
        args.use_coco_eval = False
    if not hasattr(args, 'coco_val_path'):
        args.coco_val_path = None
    if not hasattr(args, 'label_list'):
        # 设置一个默认的 label_list，避免 PostProcess 初始化时出错
        args.label_list = ['object']  # 默认类别列表

    # 构建模型
    try:
        result = build_model(args)
        # build_model 返回 (model, criterion, postprocessors) 或 model
        if isinstance(result, tuple):
            model = result[0]  # 取第一个元素（model）
        else:
            model = result
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        model.to(device)
        return model
    except Exception as e:
        import traceback
        print(f"   Detailed error: {e}")
        print(f"   Error type: {type(e).__name__}")
        traceback.print_exc()
        raise

def main():
    # ====== 3. 自动检测可用的 checkpoint 和对应的配置文件 ======
    checkpoints_dir = os.path.join(open_gdino_root, "checkpoints")
    tools_dir = os.path.join(open_gdino_root, "tools")
    
    # 定义 checkpoint 和 config 的映射关系
    checkpoint_config_map = {
        "groundingdino_swint_ogc.pth": "GroundingDINO_SwinT_OGC.py",
        "groundingdino_swinb_cogcoor.pth": "GroundingDINO_SwinB_cfg.py",
    }
    
    # 查找可用的 checkpoint
    checkpoint_path = None
    config_path = None
    
    if os.path.exists(checkpoints_dir):
        available_checkpoints = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]
        for ckpt_name in checkpoint_config_map.keys():
            ckpt_path = os.path.join(checkpoints_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                # 验证 checkpoint 文件是否完整
                if not is_checkpoint_valid(ckpt_path):
                    print(f"Warning: Checkpoint file {ckpt_name} appears to be corrupted, skipping...")
                    continue
                    
                checkpoint_path = ckpt_path
                config_name = checkpoint_config_map[ckpt_name]
                config_path = os.path.join(tools_dir, config_name)
                if os.path.exists(config_path):
                    print(f"Found checkpoint: {ckpt_name}")
                    print(f"Using config: {config_name}")
                    break
                else:
                    print(f"Warning: Config file not found for {ckpt_name}: {config_name}")
                    checkpoint_path = None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 检查 HuggingFace 缓存
    hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    bert_cache_pattern = "models--bert-base-uncased"
    has_bert_cache = False
    if os.path.exists(hf_cache_dir):
        for item in os.listdir(hf_cache_dir):
            if item.startswith(bert_cache_pattern):
                has_bert_cache = True
                print(f"Found HuggingFace cache for bert-base-uncased")
                break
    if not has_bert_cache:
        print(f"Warning: HuggingFace cache for bert-base-uncased not found in {hf_cache_dir}")
        print("  The model will try to download it from the internet.")
        print("  If you have network issues, you can copy the cache from a machine with internet access.")

    # 检查配置文件是否存在
    if config_path is None or not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        print("Available config files in tools/:")
        if os.path.exists(tools_dir):
            for f in os.listdir(tools_dir):
                if f.endswith(".py") and "GroundingDINO" in f:
                    print(f"  - {f}")
        return

    # 检查 checkpoint 是否存在
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        print(f"Warning: No valid checkpoint found in {checkpoints_dir}")
        if os.path.exists(checkpoints_dir):
            print("Available checkpoint files:")
            for f in os.listdir(checkpoints_dir):
                if f.endswith(".pth"):
                    print(f"  - {f}")
        print("You can skip model loading and test with a dummy model.")
        use_dummy = True
    else:
        use_dummy = False
        print(f"Using checkpoint: {os.path.basename(checkpoint_path)}")

    if not use_dummy:
        print("=> Loading model...")
        try:
            model = load_model(config_path, checkpoint_path, device=device)
            print("   Model loaded successfully.")
        except Exception as e:
            print(f"   Error loading model: {e}")
            print("   Trying to test import and config loading only...")
            use_dummy = True

    # ====== 4. 准备一张测试图片 ======
    # 尝试使用 figs 目录中的图片
    test_image_paths = [
        os.path.join(open_gdino_root, "figs", "dog_cat.jpeg"),
        os.path.join(open_gdino_root, "figs", "cat-1.8m.jpg"),
        os.path.join(open_gdino_root, "figs", "dog-1.8m.jpg"),
    ]
    
    image_path = None
    for path in test_image_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if image_path is None:
        print("Warning: No test image found in figs/ directory.")
        print("Available files in figs/:")
        if os.path.exists(os.path.join(open_gdino_root, "figs")):
            for f in os.listdir(os.path.join(open_gdino_root, "figs")):
                print(f"  - {f}")
        return

    print(f"=> Loading test image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    # 使用与 inference_on_a_image.py 相同的 transform
    # 直接导入 datasets.transforms
    sys.path.insert(0, os.path.join(open_gdino_root, "datasets"))
    sys.path.insert(0, os.path.join(open_gdino_root, "util"))
    import transforms as T_gdino
    transform = T_gdino.Compose([
        T_gdino.RandomResize([800], max_size=1333),
        T_gdino.ToTensor(),
        T_gdino.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor, _ = transform(image, None)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # ====== 5. 文本提示（GroundingDINO 是整句 prompt） ======
    caption = "a dog . a person . a cat ."

    if not use_dummy:
        # ====== 6. 前向推理（具体参数名字可能因 repo 略有不同） ======
        print(f"=> Running forward pass with caption: '{caption}'")
        with torch.no_grad():
            outputs = model(img_tensor, captions=[caption])

        # ====== 7. 看一下输出结构 ======
        print("\n=== Outputs structure ===")
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if torch.is_tensor(v):
                    print(f"  {k}: {v.shape} (dtype: {v.dtype})")
                elif isinstance(v, (list, tuple)):
                    print(f"  {k}: {type(v).__name__} with {len(v)} items")
                    if len(v) > 0 and torch.is_tensor(v[0]):
                        print(f"    First item shape: {v[0].shape}")
                else:
                    print(f"  {k}: {type(v).__name__}")
        else:
            print(f"Output type: {type(outputs)}")
            if torch.is_tensor(outputs):
                print(f"  Shape: {outputs.shape}")

        # 打印一些统计信息
        if "pred_logits" in outputs:
            logits = outputs["pred_logits"].sigmoid()
            max_logits = logits.max(dim=-1)[0]
            print(f"\n=== Prediction statistics ===")
            print(f"  Max logit per query: min={max_logits.min().item():.4f}, max={max_logits.max().item():.4f}, mean={max_logits.mean().item():.4f}")
            num_detections = (max_logits > 0.3).sum().item()
            print(f"  Number of detections (threshold=0.3): {num_detections}")

        print("\n✅ GroundingDINO forward pass completed successfully!")
    else:
        print("\n✅ Import and configuration loading successful!")
        print("   (Model checkpoint not available, skipping forward pass)")

if __name__ == "__main__":
    main()

