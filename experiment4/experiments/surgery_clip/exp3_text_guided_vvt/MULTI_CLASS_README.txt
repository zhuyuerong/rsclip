Multi-Class Heatmap Generator - README
=======================================

PURPOSE
=======
For images containing multiple object classes, generate separate heatmaps 
for each class with corresponding GT boxes highlighted.

EXAMPLE: DIOR_05386
===================
This image contains: 1 overpass + 3 vehicles

Generated output (2 rows x 13 columns):
Row 1: Query "an aerial photo of overpass"
  - Heatmap focuses on overpass region
  - Green boxes (thick): overpass (1 box)
  - Yellow boxes (thin): vehicle (3 boxes)

Row 2: Query "an aerial photo of vehicle"
  - Heatmap focuses on vehicle regions
  - Green boxes (thick): vehicle (3 boxes)
  - Yellow boxes (thin): overpass (1 box)

COLOR CODING
============
Green (lime, 2.5px): Query class GT boxes
Yellow (1.0px): Other class GT boxes in the same image

This helps visualize:
1. How different text queries produce different heatmaps
2. Which GT boxes correspond to the current query
3. Context of other objects in the scene

LAYOUT
======
N rows x 13 columns
- N = number of unique classes in the image
- Column 0: Original image with colored GT boxes
- Columns 1-12: Heatmaps for L1-L12

GENERATED FILES
===============
multi_class_results/
- multi_class_DIOR_03135.png (2 classes: Expressway-toll-station, vehicle)
- multi_class_DIOR_05386.png (2 classes: overpass, vehicle)
- multi_class_DIOR_09601.png (2 classes: vehicle, tenniscourt)

File sizes: ~2-3MB per image (2 rows x 13 columns)

RUNNING COMMAND
===============
cd /media/ubuntu22/新加卷1/Projects/RemoteCLIP-main

# Generate multi-class heatmaps (12 layers)
PYTHONPATH=. ovadetr_env/bin/python3.9 \
  experiment4/experiments/surgery_clip/exp3_text_guided_vvt/multi_class_heatmap.py \
  --dataset datasets/mini_dataset \
  --max-samples 5 \
  --layers 1 2 3 4 5 6 7 8 9 10 11 12

# Generate multi-class heatmaps (5 key layers)
PYTHONPATH=. ovadetr_env/bin/python3.9 \
  experiment4/experiments/surgery_clip/exp3_text_guided_vvt/multi_class_heatmap.py \
  --dataset datasets/mini_dataset \
  --max-samples 5 \
  --layers 1 3 6 9 12

CODE DETAILS
============
File: multi_class_heatmap.py (260 lines)

Key functions:
- generate_multi_class_heatmaps(): 
  * Generate heatmaps for each class in the image
  * Uses CLIP prompt: "an aerial photo of {class}"
  * Uses clip_feature_surgery() with all 20 DIOR classes
  
- visualize_multi_class_comparison():
  * One row per class
  * Green boxes for query class
  * Yellow boxes for other classes
  
- main():
  * Auto-detect multi-class images
  * Skip single-class images
  * Process unique classes only (no duplicates)

DEPENDENCIES
============
- experiment4/core/models/clip_surgery.py
  * clip_feature_surgery (line 15-59)
  * get_similarity_map (line 62-100)
  * CLIPSurgeryWrapper.get_layer_features (line 561-629)
  
- utils/seen_unseen_split.py
  * SeenUnseenDataset (returns 'classes' list)

DATASET REQUIREMENTS
====================
Dataset must return:
- 'classes': list of all class names in the image
- 'bboxes': list of bbox dicts with 'class' field
- 'original_size': (H, W) for bbox scaling

VERIFICATION
============
Check DIOR_05386:
1. Row 1 (overpass query): Green box on the bridge structure
2. Row 2 (vehicle query): Green boxes on the 3 small cars

Check DIOR_03135:
1. Row 1 (Expressway-toll-station): Green box on toll booth
2. Row 2 (vehicle): Green box on vehicle

KEY INSIGHT
===========
Same image + different text query → different heatmap
This demonstrates:
- Text-guided attention mechanism
- Class-specific feature extraction
- Surgery去冗余 removes shared features, enhances class distinction

