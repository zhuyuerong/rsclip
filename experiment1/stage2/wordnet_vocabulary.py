#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遥感图像WordNet词表
基于遥感图像常见场景构建的通用词表（100类）
涵盖：建筑、交通、自然、城市、工业、农业、舰船、飞行器、车辆
"""

# 遥感图像通用词表（涵盖主要遥感场景）
WORDNET_REMOTE_SENSING_CLASSES = [
    # ===== 建筑物类 (Buildings) - 15类 =====
    "residential building", "commercial building", "industrial building",
    "apartment", "house", "villa", "warehouse", "factory",
    "school", "hospital", "hotel", "stadium", "church",
    "office building", "shopping mall",
    
    # ===== 交通设施类 (Transportation Infrastructure) - 15类 =====
    "airport", "runway", "taxiway", "apron", "terminal",
    "highway", "road", "street", "bridge", "overpass",
    "railway", "railway station", "port", "harbor", "parking lot",
    
    # ===== 自然地物类 (Natural Features) - 15类 =====
    "forest", "woodland", "grassland", "meadow", "farmland",
    "cropland", "orchard", "vineyard", "lake", "river",
    "pond", "stream", "mountain", "hill", "desert",
    
    # ===== 城市设施类 (Urban Facilities) - 15类 =====
    "park", "playground", "sports field", "tennis court", "basketball court",
    "swimming pool", "golf course", "cemetery", "power plant", "solar farm",
    "wind farm", "substation", "water treatment plant", "landfill", "quarry",
    
    # ===== 工业设施类 (Industrial) - 10类 =====
    "storage tank", "oil tank", "gas station", "refinery", "chimney",
    "crane", "construction site", "excavation", "mining area", "dump truck",
    
    # ===== 农业设施类 (Agriculture) - 5类 =====
    "greenhouse", "barn", "silo", "irrigation system", "fish pond",
    
    # ===== 舰船类 (Ships & Vessels) - 10类 =====
    "ship", "vessel", "warship", "cargo ship", "tanker ship",
    "fishing boat", "yacht", "aircraft carrier", "naval vessel", "boat",
    
    # ===== 飞行器类 (Aircraft) - 10类 =====
    "airplane", "aircraft", "helicopter", "jet", "fighter jet",
    "transport plane", "passenger aircraft", "military aircraft", "drone", "glider",
    
    # ===== 车辆类 (Vehicles) - 5类 =====
    "vehicle", "car", "truck", "bus", "train"
]

# 保持向后兼容
WORDNET_80_CLASSES = WORDNET_REMOTE_SENSING_CLASSES

# WordNet分类层次结构（用于扩展词生成）
WORDNET_HIERARCHY = {
    # 建筑物上位词
    "building": ["residential building", "commercial building", "industrial building", 
                 "apartment", "house", "villa", "warehouse", "factory",
                 "school", "hospital", "hotel", "stadium", "church",
                 "office building", "shopping mall"],
    
    # 交通设施上位词
    "transportation infrastructure": ["airport", "runway", "taxiway", "apron", "terminal",
                                      "highway", "road", "street", "bridge", "overpass",
                                      "railway", "railway station", "port", "harbor"],
    
    # 自然地物上位词
    "natural feature": ["forest", "woodland", "grassland", "meadow",
                        "lake", "river", "pond", "stream",
                        "mountain", "hill", "desert"],
    
    # 农业用地上位词
    "agricultural land": ["farmland", "cropland", "orchard", "vineyard",
                          "greenhouse", "barn", "silo", "irrigation system", "fish pond"],
    
    # 运动设施上位词
    "sports facility": ["sports field", "tennis court", "basketball court",
                        "swimming pool", "golf course", "playground"],
    
    # 工业设施上位词
    "industrial facility": ["storage tank", "oil tank", "gas station", "refinery",
                           "power plant", "solar farm", "wind farm",
                           "construction site", "mining area"],
    
    # 交通工具上位词
    "vehicle": ["car", "truck", "bus", "ship", "airplane", "train", "container"],
    
    # 舰船上位词
    "vessel": ["ship", "warship", "cargo ship", "tanker ship", "fishing boat", 
                "yacht", "aircraft carrier", "naval vessel", "boat"],
    
    # 飞行器上位词
    "aircraft": ["airplane", "helicopter", "jet", "fighter jet", "transport plane",
                 "passenger aircraft", "military aircraft", "drone", "glider"],
    
    # 地面车辆上位词
    "ground vehicle": ["vehicle", "car", "truck", "bus"],
    
    # 城市设施上位词
    "urban facility": ["park", "cemetery", "water treatment plant", "landfill", "quarry"]
}

# 近义词映射（用于扩展）- 遥感通用
WORDNET_SYNONYMS = {
    # 建筑设施
    "building": ["structure", "edifice", "construction", "premises"],
    "road": ["highway", "street", "pathway", "route", "roadway"],
    "facility": ["infrastructure", "installation", "plant", "complex"],
    
    # 自然地物
    "water": ["lake", "river", "pond", "stream", "waterbody"],
    "vegetation": ["forest", "woodland", "grassland", "meadow", "greenery"],
    "mountain": ["hill", "peak", "highland", "elevation"],
    
    # 舰船相关
    "ship": ["vessel", "boat", "watercraft", "marine vehicle"],
    "warship": ["naval vessel", "military ship", "warcraft", "combat ship"],
    "cargo ship": ["freighter", "merchant ship", "transport vessel"],
    "tanker": ["tanker ship", "oil tanker", "bulk carrier"],
    
    # 飞行器相关
    "airplane": ["aircraft", "plane", "aeroplane", "flying vehicle"],
    "helicopter": ["chopper", "rotorcraft", "whirlybird"],
    "jet": ["jet aircraft", "jetliner", "jet plane"],
    "drone": ["UAV", "unmanned aerial vehicle", "quadcopter"],
    
    # 车辆相关
    "vehicle": ["automobile", "motor vehicle", "transport"],
    "car": ["automobile", "sedan", "vehicle"],
    "truck": ["lorry", "heavy vehicle", "cargo truck"],
}


def get_hypernyms(class_name):
    """
    获取给定类别的上位词
    
    参数:
        class_name: 类别名称
    
    返回:
        上位词列表
    """
    hypernyms = []
    for hypernym, subclasses in WORDNET_HIERARCHY.items():
        if class_name in subclasses:
            hypernyms.append(hypernym)
    return hypernyms


def get_synonyms(class_name):
    """
    获取给定类别的近义词
    
    参数:
        class_name: 类别名称
    
    返回:
        近义词列表
    """
    # 直接查找
    if class_name in WORDNET_SYNONYMS:
        return WORDNET_SYNONYMS[class_name]
    
    # 查找部分匹配
    synonyms = []
    for key, values in WORDNET_SYNONYMS.items():
        if key in class_name or class_name in key:
            synonyms.extend(values)
    
    return synonyms


def get_expansion_words(unseen_class, num_words=5):
    """
    为未见类别生成扩展词
    
    参数:
        unseen_class: 未见过的类别名称
        num_words: 需要生成的扩展词数量
    
    返回:
        扩展词列表
    """
    expansion = []
    
    # 1. 获取上位词
    hypernyms = get_hypernyms(unseen_class)
    expansion.extend(hypernyms)
    
    # 2. 获取近义词
    synonyms = get_synonyms(unseen_class)
    expansion.extend(synonyms)
    
    # 3. 如果还不够，添加通用描述
    if len(expansion) < num_words:
        generic_terms = [
            "object", "structure", "area", "feature", "element"
        ]
        expansion.extend(generic_terms)
    
    # 去重并限制数量
    expansion = list(dict.fromkeys(expansion))  # 保持顺序去重
    return expansion[:num_words]


def build_vocabulary(unseen_class=None):
    """
    构建完整的词表
    
    参数:
        unseen_class: 未见过的类别（可选）
    
    返回:
        完整词表字典，包含：
        - base_classes: 80个基础类别
        - expansion_words: 5个扩展词（如果提供unseen_class）
        - unseen: 未见类别标签
        - total: 总类别数
    """
    vocab = {
        'base_classes': WORDNET_80_CLASSES.copy(),
        'expansion_words': [],
        'unseen_label': 'unseen',
        'total_classes': 80
    }
    
    if unseen_class:
        expansion = get_expansion_words(unseen_class, num_words=5)
        vocab['expansion_words'] = expansion
        vocab['unseen_class'] = unseen_class
        vocab['total_classes'] = 80 + 5 + 1  # 80 + 5扩展 + 1未见
    
    return vocab


def get_full_class_list(unseen_class=None):
    """
    获取完整的类别列表（用于RemoteCLIP推理）
    
    参数:
        unseen_class: 未见过的类别（可选）
    
    返回:
        类别列表
    """
    classes = WORDNET_80_CLASSES.copy()
    
    if unseen_class:
        expansion = get_expansion_words(unseen_class, num_words=5)
        classes.extend(expansion)
        classes.append('unseen')
    
    return classes


def print_vocabulary_info(unseen_class=None):
    """打印词表信息"""
    vocab = build_vocabulary(unseen_class)
    
    print("=" * 70)
    print("遥感图像通用WordNet词表")
    print("=" * 70)
    
    print(f"\n基础类别数: {len(vocab['base_classes'])}")
    print("\n类别分布:")
    print("  - 建筑物类: 15")
    print("  - 交通设施类: 15")
    print("  - 自然地物类: 15")
    print("  - 城市设施类: 15")
    print("  - 工业设施类: 10")
    print("  - 农业设施类: 5")
    print("  - 舰船类: 10 ⭐")
    print("  - 飞行器类: 10 ⭐")
    print("  - 车辆类: 5")
    
    if unseen_class:
        print(f"\n未见类别: {unseen_class}")
        print(f"扩展词数: {len(vocab['expansion_words'])}")
        print(f"扩展词: {vocab['expansion_words']}")
    
    print(f"\n总类别数: {vocab['total_classes']}")
    print("=" * 70)


if __name__ == "__main__":
    # 测试1: 基础词表
    print_vocabulary_info()
    
    # 测试2: 带未见类别的词表
    print("\n")
    print_vocabulary_info(unseen_class="wind turbine")
    
    # 测试3: 获取完整类别列表
    print("\n完整类别列表示例（前20个）:")
    classes = get_full_class_list()
    for i, cls in enumerate(classes[:20], 1):
        print(f"{i:2d}. {cls}")

