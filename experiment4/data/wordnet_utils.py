# -*- coding: utf-8 -*-
"""
WordNet辅助词表构建
为每个类别生成相关词汇
"""

import random


# 预定义的遥感类别词表（基于mini_dataset的20个类别）
WORDNET_DICT = {
    'airplane': [
        'aircraft', 'plane', 'jet', 'airliner', 'fighter',
        'aviation', 'flying machine', 'air vehicle', 'jetliner',
        'cargo plane', 'military aircraft', 'commercial aircraft',
        'propeller plane', 'wing', 'fuselage', 'cockpit',
        'aircraft on ground', 'parked airplane', 'flying airplane',
        'airplane from above'
    ],
    'airport': [
        'airfield', 'aerodrome', 'landing field', 'runway',
        'terminal', 'control tower', 'taxiway', 'apron',
        'aviation facility', 'aircraft parking', 'hangar',
        'airport building', 'aviation infrastructure', 'flight zone',
        'air traffic area', 'landing strip', 'airport terminal',
        'airport complex', 'aviation hub', 'aerial view of airport'
    ],
    'baseballfield': [
        'baseball diamond', 'ball field', 'sports field', 'diamond',
        'infield', 'outfield', 'baseball stadium', 'ballpark',
        'baseball ground', 'sports facility', 'athletic field',
        'baseball arena', 'baseball complex', 'sports venue',
        'baseball field from above', 'diamond shape', 'baseball pitch',
        'recreational field', 'sports ground', 'baseball site'
    ],
    'basketballcourt': [
        'basketball field', 'court', 'sports court', 'basketball ground',
        'basketball arena', 'basketball facility', 'recreational court',
        'sports surface', 'basketball site', 'basketball venue',
        'hardcourt', 'outdoor court', 'indoor court', 'basketball floor',
        'basketball complex', 'basketball pitch', 'sports infrastructure',
        'athletic court', 'basketball area', 'court from above'
    ],
    'beach': [
        'seashore', 'coast', 'shore', 'seaside', 'coastline',
        'waterfront', 'sandy beach', 'shoreline', 'beachfront',
        'coastal area', 'sand', 'ocean edge', 'sea edge',
        'beach sand', 'coastal zone', 'littoral', 'strand',
        'beach area', 'coastal region', 'beach from above'
    ],
    'bridge': [
        'overpass', 'viaduct', 'crossing', 'span', 'flyover',
        'road bridge', 'railway bridge', 'footbridge', 'arch bridge',
        'suspension bridge', 'bridge structure', 'river crossing',
        'bridge deck', 'bridge from above', 'aerial bridge',
        'bridge infrastructure', 'transportation bridge', 'water crossing',
        'bridge construction', 'engineering structure'
    ],
    'chaparral': [
        'shrubland', 'bushland', 'scrubland', 'vegetation',
        'dry vegetation', 'mediterranean vegetation', 'shrub',
        'wild vegetation', 'natural vegetation', 'plant cover',
        'shrub area', 'bushy area', 'wilderness', 'natural area',
        'vegetation cover', 'arid vegetation', 'dry bushland',
        'chaparral ecosystem', 'shrub ecosystem', 'natural landscape'
    ],
    'church': [
        'cathedral', 'chapel', 'temple', 'religious building',
        'place of worship', 'sanctuary', 'basilica', 'church building',
        'religious structure', 'worship place', 'church architecture',
        'church from above', 'religious site', 'church roof',
        'church complex', 'christian building', 'parish church',
        'church tower', 'church steeple', 'religious facility'
    ],
    'circularfarmland': [
        'circular field', 'pivot irrigation', 'round farmland',
        'irrigation circle', 'agricultural circle', 'farm circle',
        'center pivot', 'circular crop', 'circular agriculture',
        'round field', 'irrigated farmland', 'pivot field',
        'circular cultivation', 'round cultivation', 'farm pattern',
        'agricultural pattern', 'irrigation pattern', 'circular farm',
        'pivot irrigation system', 'agricultural field'
    ],
    'cloud': [
        'clouds', 'sky cover', 'cloud cover', 'cloud formation',
        'white cloud', 'cumulus', 'weather', 'atmosphere',
        'cloud patch', 'cloud area', 'cloud from above',
        'cloud pattern', 'atmospheric cloud', 'sky clouds',
        'cloud layer', 'cloud mass', 'cloud structure',
        'meteorological cloud', 'cloudy area', 'aerial cloud'
    ],
    'denseresidential': [
        'urban area', 'residential area', 'dense housing',
        'crowded residential', 'housing complex', 'urban residential',
        'dense neighborhood', 'residential district', 'housing area',
        'urban housing', 'residential zone', 'dense urban',
        'crowded housing', 'residential buildings', 'urban settlement',
        'housing density', 'residential development', 'urban development',
        'dense buildings', 'residential cluster'
    ],
    'forest': [
        'woodland', 'woods', 'trees', 'tree cover', 'forest area',
        'tree canopy', 'dense trees', 'forest cover', 'forested area',
        'tree cluster', 'natural forest', 'forest vegetation',
        'tree vegetation', 'wooded area', 'forest from above',
        'forest canopy', 'tree tops', 'forest pattern',
        'forest landscape', 'green forest'
    ],
    'freeway': [
        'highway', 'expressway', 'motorway', 'interstate',
        'road', 'major road', 'multi-lane road', 'highway from above',
        'transportation corridor', 'road infrastructure', 'highway system',
        'freeway interchange', 'road network', 'highway network',
        'transportation route', 'expressway system', 'road from above',
        'highway aerial', 'freeway aerial', 'road aerial'
    ],
    'golfcourse': [
        'golf field', 'golf ground', 'golf facility', 'golf club',
        'golf course green', 'fairway', 'putting green', 'golf links',
        'golf complex', 'golf site', 'recreational golf', 'golf venue',
        'golf course from above', 'golf terrain', 'golf landscape',
        'golf area', 'golf ground pattern', 'golf course pattern',
        'golf facility aerial', 'golf course aerial'
    ],
    'harbor': [
        'port', 'seaport', 'marina', 'dock', 'wharf',
        'harbor area', 'port facility', 'shipping port', 'boat harbor',
        'harbor from above', 'port from above', 'harbor structure',
        'port infrastructure', 'maritime facility', 'harbor complex',
        'port complex', 'shipping facility', 'harbor aerial',
        'port aerial', 'maritime port'
    ],
    'intersection': [
        'road intersection', 'crossroads', 'junction', 'road crossing',
        'street intersection', 'road junction', 'crossing point',
        'intersection from above', 'road meeting', 'traffic intersection',
        'street crossing', 'road crossroads', 'intersection aerial',
        'junction aerial', 'road network node', 'street junction',
        'intersection pattern', 'crossroads from above', 'traffic junction',
        'road node'
    ],
    'mediumresidential': [
        'residential area', 'suburban area', 'housing area',
        'residential zone', 'suburban residential', 'medium density housing',
        'residential district', 'housing district', 'suburban housing',
        'residential neighborhood', 'medium residential zone',
        'housing neighborhood', 'residential from above', 'suburban from above',
        'residential pattern', 'housing pattern', 'suburban pattern',
        'medium density area', 'residential development', 'suburban development'
    ],
    'mobilehomepark': [
        'mobile home area', 'trailer park', 'manufactured housing',
        'mobile housing', 'trailer area', 'mobile home community',
        'manufactured home park', 'trailer community', 'mobile home site',
        'mobile home from above', 'trailer park from above', 'mobile housing area',
        'manufactured housing area', 'mobile home pattern', 'trailer park pattern',
        'mobile home complex', 'trailer park complex', 'mobile housing complex',
        'manufactured home area', 'mobile home aerial'
    ],
    'overpass': [
        'bridge', 'flyover', 'road overpass', 'highway overpass',
        'elevated road', 'overpass structure', 'road bridge',
        'highway bridge', 'overpass from above', 'elevated crossing',
        'overpass aerial', 'road crossing structure', 'highway crossing',
        'overpass infrastructure', 'elevated roadway', 'overpass construction',
        'road overbridge', 'highway overbridge', 'overpass system',
        'transportation overpass'
    ],
    'parkinglot': [
        'parking area', 'car park', 'parking space', 'vehicle parking',
        'parking facility', 'parking ground', 'parking lot from above',
        'parking aerial', 'parking lot pattern', 'car parking',
        'vehicle lot', 'parking structure', 'parking complex',
        'parking zone', 'parking site', 'parking area from above',
        'parking lot aerial', 'car park aerial', 'parking infrastructure',
        'vehicle parking area'
    ]
}


def get_wordnet_words(class_name, k=20):
    """
    获取类别的WordNet相关词
    
    Args:
        class_name: 类别名
        k: 返回词数
    
    Returns:
        words: list of str
    """
    # 归一化类别名
    class_name_lower = class_name.lower().replace(' ', '').replace('-', '').replace('_', '')
    
    # 查找匹配的词表
    if class_name_lower in WORDNET_DICT:
        words = WORDNET_DICT[class_name_lower].copy()
    else:
        # 如果没有预定义，生成默认词表
        words = generate_default_words(class_name)
    
    # 确保包含原始类别名
    if class_name not in words:
        words.insert(0, class_name)
    
    # 如果词数不够，重复采样
    if len(words) < k:
        words = words + random.sample(words, k - len(words))
    
    # 返回前k个
    return words[:k]


def generate_default_words(class_name):
    """
    为未知类别生成默认词表
    
    Args:
        class_name: 类别名
    
    Returns:
        words: list of str
    """
    words = [
        class_name,
        f"{class_name} from above",
        f"{class_name} aerial view",
        f"{class_name} satellite image",
        f"{class_name} aerial",
        f"{class_name} from top",
        f"{class_name} bird view",
        f"{class_name} overhead view",
        f"{class_name} area",
        f"{class_name} region",
        f"{class_name} zone",
        f"{class_name} site",
        f"{class_name} location",
        f"{class_name} place",
        f"{class_name} structure",
        f"{class_name} pattern",
        f"{class_name} feature",
        f"{class_name} object",
        f"{class_name} target",
        f"{class_name} entity"
    ]
    
    return words


def get_all_classes_words(class_names, k=20):
    """
    获取所有类别的词表
    
    Args:
        class_names: list of str
        k: 每个类别的词数
    
    Returns:
        words_dict: {class_name: [words]}
    """
    words_dict = {}
    
    for class_name in class_names:
        words_dict[class_name] = get_wordnet_words(class_name, k)
    
    return words_dict


def test_wordnet():
    """测试WordNet工具"""
    print("测试WordNet工具...")
    
    # 测试几个类别
    test_classes = ['airplane', 'ship', 'car', 'unknown_class']
    
    for cls in test_classes:
        words = get_wordnet_words(cls, k=10)
        print(f"\n{cls}: {words[:5]}...")
    
    print("\n测试通过！")


if __name__ == "__main__":
    test_wordnet()

