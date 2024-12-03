import argparse
from motionepic.model.builder import load_pretrained_model
from motionepic.mm_utils import get_model_name_from_path


def merge_lora(args):
    # model_name = get_model_name_from_path(args.model_path) 
    model_name = 'motionepic-v1.5-7b-lora'
    tokenizer, model, video_processor, context_len, model_config = load_pretrained_model(args.model_path, args.model_base, model_name, device_map='cpu')

    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default='./checkpoints/finetune')
    parser.add_argument("--model-base", type=str, default='./checkpoints/pretrain')
    parser.add_argument("--save-model-path", type=str, default='./checkpoints/motionepic-v1.5-7b-lora')

    args = parser.parse_args()

    merge_lora(args)