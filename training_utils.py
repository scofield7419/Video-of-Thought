from dataclasses import dataclass, field
from typing import List, Optional
from transformers import TrainingArguments


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )

    # LoRA related parameters
    lora_enable: bool = field(default=False, metadata={"help": "Whether to use LoRA technique"})
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_input_projector_lr: Optional[float] = None
    mm_output_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    group_by_modality_type: bool = field(default=False)

    fine_tune: bool = field(default=False, metadata={"help": "Whether to fine-tune the model."})
    freeze_mm_input_adapter: bool = field(default=False)



@dataclass
class DataArguments:
    dataset_name_list: List[str] = field(default=None, metadata={"help": "The list of dataset names"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    video_folder: Optional[str] = field(default=None)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=None, metadata={"help": "Build-in pretrained model name or the path to local model."}
    )
    version: Optional[str] = field(default="v0")
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    mm_use_vid_start_end: bool = field(default=False)
    mm_use_vid_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")

    version: Optional[str] = field(default="v0")
    multimodal_tower: Optional[str] = field(default=None)
    freeze_backbone: bool = field(default=True)
    tune_mm_input_adapter: bool = field(default=True)
    pretrain_mm_input_adapter: Optional[str] = field(default=None)
    mm_input_projector_type: Optional[str] = field(default='linear')

    # STSG related parameters
    stsg_num_heads: int = field(default=8)
    stsg_in_dim: int = field(default=512)
    stsg_out_dim: int = field(default=512)
    stsg_num_layers: int = field(default=3)
    stsg_dropout: float = field(default=0.0)

    num_query_token: int = field(default=64)

    


