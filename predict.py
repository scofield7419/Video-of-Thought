import torch

from motionepic.constants import DEFAULT_VIDEO_TOKEN, DEFAULT_SG_TOKEN
from motionepic.conversation import conv_templates, SeparatorStyle
from motionepic.model.builder import load_pretrained_model
from motionepic.utils import disable_torch_init
from motionepic.mm_utils import tokenizer_multiple_token
from transformers.generation.streamers import TextIteratorStreamer
import transformers
from dataclasses import dataclass, field
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList
from typing import List

import requests
from io import BytesIO
import scipy
from cog import BasePredictor, Input, Path, ConcatenateIterator
import time
import subprocess
from threading import Thread
from diffusers.utils import export_to_video
import os
import random
import cv2
import numpy as np
os.environ["HUGGINGFACE_HUB_CACHE"] = os.getcwd() + "/weights"

def read_video(video_path, sample_fps=1, max_frames=8, height=320, width=576, get_first_frame=False):
    """
    Read video frames from video_path.
    Args:
        video_path: str, path to the video file.
        sample_fps: int, sample frames per second.
        max_frames: int, maximum number of frames to sample.
    Returns:
        torch.Tensor, (num_frames, channel, height, width).
    """
    height = 0
    width = 0
    for _ in range(5):
        try:
            capture = cv2.VideoCapture(video_path)
            _fps = capture.get(cv2.CAP_PROP_FPS)
            # print(_fps)
            _total_frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            # print(_total_frame_num)
            stride = round(_fps / sample_fps)
            cover_frame_num = (stride * max_frames)
            # print(cover_frame_num)
            if _total_frame_num < cover_frame_num + 5:
                start_frame = 0
                end_frame = _total_frame_num
            else:
                start_frame = random.randint(0, _total_frame_num-cover_frame_num-5)
                end_frame = start_frame + cover_frame_num
            
            pointer, frame_list = 0, []
            while(True):
                ret, frame = capture.read()
                pointer +=1 
                if (not ret) or (frame is None): break
                if pointer < start_frame: continue
                if pointer >= end_frame - 1: break
                if (pointer - start_frame) % stride == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    height, width = frame.size
                    frame_list.append(frame)
            break
        except Exception as e:
            print('{} read video frame failed with error: {}'.format(video_path, e))
            continue
    
    assert height > 0 and width > 0, "Video height and width should be greater than 0."
    # video_data = 

    dummy_frame = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    try:
        if len(frame_list)> max_frames:
            # mid_frame = copy(frame_list[ref_idx])
            # vit_frame = self.vit_transforms(mid_frame)
            # frames = self.transforms(frame_list)
            frame_list = frame_list[:max_frames]
        elif 0< len(frame_list) < max_frames:
            frame_list.extend([dummy_frame] * (max_frames - len(frame_list)))
        else:
            pass
    except Exception as e:
        print('{} read video frame failed with error: {}'.format(video_path, e))
    
    return frame_list


@dataclass
class GenerateArguments:
    # Basic generation arguments
    top_k: int = field(default=1, metadata={"help": "The number of highest probability tokens to keep for top-k-filtering in the sampling strategy"})
    top_p: float = field(default=1.0, metadata={"help": "The cumulative probability for top-p-filtering in the sampling strategy."})
    temperature: float = field(default=1.0, metadata={"help": "The value used to module the next token probabilities. Must be strictly positive."},)
    max_new_tokens: int = field(default=100, metadata={"help": "The maximum number of new tokens to generate. The generation process will stop when reaching this threshold."})
    do_sample: bool = field(default=True, metadata={"help": "Whether to sample from the output distribution to generate new tokens. If False, use argmax."})
    use_cache: bool = field(default=False, metadata={"help": "Whether to cache the hidden states of the model to speed up generation."})
    output_hidden_states: bool = field(default=True,metadata={"help": "Whether to return the hidden states of all intermediate layers."})
    


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops: List = None, encounters: int = 1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        stop_count = 0
        for stop in self.stops:
            _stop = torch.tensor(stop).to(input_ids[0].device)
            indices = torch.where(_stop[0] == input_ids)
            for i in indices:
                if len(i) > 0:
                    if torch.all(input_ids[0][i:i + len(_stop)] == _stop):
                        stop_count += 1
        if stop_count >= self.ENCOUNTERS:
            return True
        return False


def download_json(url: str, dest: Path):
    res = requests.get(url, allow_redirects=True)
    if res.status_code == 200 and res.content:
        with dest.open("wb") as f:
            f.write(res.content)
    else:
        print(f"Failed to download {url}. Status code: {res.status_code}")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        disable_torch_init()

        self.tokenizer, self.model, self.video_processor, self.context_len, self.model_config = load_pretrained_model(model_base="./checkpoints/pretrain/model", 
                                                                                                   model_name="motionepic-v1.5-7b", 
                                                                                                   model_path="./checkpoints/finetune", 
                                                                                                   load_8bit=False, load_4bit=False)

    def predict(
        self,
        video: str = None,
        prompt: str = None,
        top_p: float = 1.0,
        temperature: float = 0.2,
        max_new_tokens: int = 512,
    ):
        """Run a single prediction on the model"""

        # prepare generation arguments
        parser = transformers.HfArgumentParser(GenerateArguments)
        generation_args = parser.parse_args_into_dataclasses()[0]

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=[[835]], encounters=1)])

        generation_args.top_p = top_p if top_p is not None else generation_args.top_p
        generation_args.temperature = temperature if temperature is not None else generation_args.temperature
        generation_args.max_new_tokens = max_new_tokens if max_new_tokens is not None else generation_args.max_new_tokens
        generation_args.stopping_criteria = stopping_criteria

        if video is not None:
            video_data = read_video(str(video))
            video_tensor = torch.stack([self.video_processor(v, return_tensors='pt')['pixel_values'].half().cuda() for v in video_data], dim=0)
        else:
            video_tensor = None
  
        input_ids = tokenizer_multiple_token(prompt, self.tokenizer, return_tensors='pt').unsqueeze(0).cuda()
        
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                videos=video_tensor,
                **generation_args.__dict__
            )
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
    

    def step_1_predict(self, video: str = None, questions: str = None, is_multi_choice: bool = False):
        """
        Task Definition and Target Identification for Multi-choice Question or Open-ended Question

        """
        if is_multi_choice:
            task_definition = "Now you are an expert in analyzing video data, and you should answer a question based on the given video. For the question, several candidate answers are provided, where you need to choose [the most suitable option — all possible correct option(s)]."
        else:
            task_definition = "Now you are an expert in analyzing video data, and you should answer a question based on the given video. For the question, you should answer in an open-ended format."
        
        conv_mode = "vicuna_v1"  # conv_llava_plain  conv_vicuna_v1
        conv = conv_templates[conv_mode].copy()
        conv.system = task_definition
        prompt = f"{DEFAULT_VIDEO_TOKEN} \n Given the question: {questions}, what are the possible targets of the video mainly mentioned or involved?"
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        targets = self.predict(video=video, prompt=prompt)
        return targets
    
    def step_2_predict(self, video: str = None, targets: str = None):
        """
        Object Tracking
        """
        conv_mode = "vicuna_v1"  # conv_llava_plain  conv_vicuna_v1
        conv = conv_templates[conv_mode].copy()
        prompt = f"{DEFAULT_VIDEO_TOKEN} \n {DEFAULT_SG_TOKEN} \n Provide the tracklet of involved {' '.join(targets)} by outputting the corresponding partial spatial-temporal scene graph expression in the video."
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stsgs = self.predict(video=video, prompt=prompt)
        return stsgs
    
    def step_3_predict(self, stsgs: str = None, targets: str = None):
        """
        Action Analyzing
        """
        conv_mode = "vicuna_v1"  # conv_llava_plain  conv_vicuna_v1
        conv = conv_templates[conv_mode].copy()
        prompt = f"{DEFAULT_SG_TOKEN} \n Combining all possible related commonsense, analyze the motion behavior based on the {' '.join(targets)} and the neighbor scenes within spatial-temporal scene graph. Describing the action and the implication"        
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        actions = self.predict(prompt=prompt)
        return actions
    
    def step_4_predict(self, video: str = None, questions: str = None, candidata_answers: List[str] = None, actions: str = None, is_multi_choice: bool = False):
        """
        1) Transforming Open-ended Question Answering into Multi-choice one
        2) Multi-choice Question Answering via Ranking
        """
        if not is_multi_choice:
            conv_mode = "vicuna_v1"  # conv_llava_plain  conv_vicuna_v1
            conv = conv_templates[conv_mode].copy()
            prompt = f"For the question {questions}, please based on the action's {actions} combined with commonsense, output 4 distinct optional answers with the rationality score of this answer with a 1-10 scale."
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            candidate_answers = self.predict(prompt=prompt)
        else:
            conv_mode = "vicuna_v1"  # conv_llava_plain  conv_vicuna_v1
            conv = conv_templates[conv_mode].copy()
            prompt = ""
            for i, candidate_answer in enumerate(candidata_answers):
                prompt += f"For the question {questions}, given a candidate answer {candidate_answer}, please based on the action's {actions} combined with commonsense, score the rationality of this answer with a 1-10 scale, and also output the rationale."
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            rationality_list = self.predict(prompt=prompt)

            conv.clear()
            prompt = "Now, we know the rationale score of the answer "
            for i, (candidate_answer, rationality) in enumerate(candidata_answers, rationality_list):
                prompt += f"{candidate_answer} is {rationality_list[i]}. The answer "
            prompt += " Please rank the candidate the answer based on the rationale score of each candidate's answer."
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            final_answer = self.predict(prompt=prompt)
        return final_answer
    
    def step_5_predict(self, video: str = None, questions: str = None, answer: str = None, actions: str = None):
        """
        Answer Verification
        """
        conv_mode = "vicuna_v1"  # conv_llava_plain  conv_vicuna_v1
        conv = conv_templates[conv_mode].copy()
        prompt = f"{DEFAULT_VIDEO_TOKEN} \n Given the video and raw question {questions}, now you need to verify the previous answer by 1) checking the pixel grounding information if the answer {answer} aligns with the facts presented in the video from a perception standpoint; 2) determining from a cognition perspective if the commonsense implications inherent in the answer {answer} contradict any of the main {actions} inferred in the 3-rd reasoning step. Output the verification result with rationale."
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        answer = self.predict(prompt=prompt, video=video)
        return answer

    def cot_reasoning(self, video: str = None, questions: str = None, candidata_answers: List[str] = None, is_multi_choice: bool = False):
        targets = self.step_1_predict(video=video, questions=questions, is_multi_choice=is_multi_choice)
        stsgs = self.step_2_predict(video=video, targets=targets)
        actions = self.step_3_predict(stsgs=stsgs, targets=targets)
        final_answer = self.step_4_predict(video=video, questions=questions, candidata_answers=candidata_answers, actions=actions, is_multi_choice=is_multi_choice)
        verification = self.step_5_predict(video=video, questions=questions, answer=final_answer, actions=actions)
        return verification 

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


if __name__ == "__main__":
    predictor = Predictor()
    predictor.setup()

