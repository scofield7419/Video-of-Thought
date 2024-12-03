import os
from .clip_encoder import CLIPVisionTower
from .stsg import SpatialTemporalGraphTransformer

def build_multimodal_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'multimodal_input_tower', getattr(vision_tower_cfg, 'multimodal_tower', None))
    print(f'Building multimodal tower: {vision_tower}')
    # is_absolute_path_exists = os.path.exists(vision_tower)
    if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        print(f'Building CLIPVisionTower ... ')
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        return ValueError(f'Unknown vision tower: {vision_tower}')


def build_sg_encoder(sg_cfg, **kwargs):
    return SpatialTemporalGraphTransformer(sg_cfg, **kwargs)