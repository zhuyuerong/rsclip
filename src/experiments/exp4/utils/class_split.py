# -*- coding: utf-8 -*-
"""
Seen/Unseen类别划分
用于开放词汇检测实验
"""

# DIOR全部20个类别
ALL_CLASSES = [
    "airplane", "airport", "baseball field", "basketball court",
    "bridge", "chimney", "dam", "expressway service area",
    "expressway toll station", "golf course", "ground track field",
    "harbor", "overpass", "ship", "stadium", "storage tank",
    "tennis court", "train station", "vehicle", "wind mill"
]

# Seen类别（训练时可见的10个类别）
SEEN_CLASSES = [
    "airplane", "ship", "vehicle", "bridge", "harbor",
    "stadium", "storage tank", "airport", "golf course", "wind mill"
]

# Unseen类别（训练时不可见的10个类别）
UNSEEN_CLASSES = [
    "baseball field", "basketball court", "chimney", "dam",
    "expressway service area", "expressway toll station",
    "ground track field", "overpass", "tennis court", "train station"
]

def get_seen_class_indices():
    """获取seen类别的索引列表"""
    return [ALL_CLASSES.index(cls) for cls in SEEN_CLASSES]

def get_unseen_class_indices():
    """获取unseen类别的索引列表"""
    return [ALL_CLASSES.index(cls) for cls in UNSEEN_CLASSES]

def is_seen_class(class_name: str) -> bool:
    """判断类别是否为seen类别"""
    return class_name.lower() in [c.lower() for c in SEEN_CLASSES]

def is_unseen_class(class_name: str) -> bool:
    """判断类别是否为unseen类别"""
    return class_name.lower() in [c.lower() for c in UNSEEN_CLASSES]

def filter_seen_classes(boxes, labels, class_names):
    """
    过滤出seen类别的标注
    
    Args:
        boxes: List of boxes
        labels: List of label indices
        class_names: List of class names
    
    Returns:
        filtered_boxes, filtered_labels
    """
    seen_indices = get_seen_class_indices()
    filtered_boxes = []
    filtered_labels = []
    
    for box, label in zip(boxes, labels):
        if label in seen_indices:
            filtered_boxes.append(box)
            filtered_labels.append(label)
    
    return filtered_boxes, filtered_labels


