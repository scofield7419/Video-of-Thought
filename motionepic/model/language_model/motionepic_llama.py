# Copyright (c) 2024 torchtorch Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
import warnings
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, GenerationConfig

from transformers.modeling_outputs import ModelOutput
from transformers.generation.utils import GenerateOutput

from ..motionepic_arch import MotionEpicMetaForCausalLM, MotionEpicMetaModel
from transformers import StoppingCriteria, StoppingCriteriaList


@dataclass
class CausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
   

__all__ = [
    "MotionEpicLlamaModel",
    "MotionEpicLlamaForCausalLM",
]



class MotionEpicConfig(LlamaConfig):
    model_type = "motionepic_llama"


class MotionEpicLlamaModel(MotionEpicMetaModel, LlamaModel):
    config_class = MotionEpicConfig

    def __init__(self, config: LlamaConfig):
        super(MotionEpicLlamaModel, self).__init__(config)


class MotionEpicLlamaForCausalLM(LlamaForCausalLM, MotionEpicMetaForCausalLM):
    config_class = MotionEpicConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MotionEpicLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.config = config

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def _get_output(
            self,
            input_ids: torch.Tensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.Tensor]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = True,
            output_hidden_states: Optional[bool] = True,
            cache_position: Optional[torch.Tensor] = None,
            images: Union[Optional[torch.Tensor], Optional[List[torch.Tensor]]] = None,
            videos: Union[Optional[torch.Tensor], Optional[List[torch.Tensor]]] = None,
            audios: Union[Optional[torch.Tensor], Optional[List[torch.Tensor]]] = None,
            return_dict: Optional[bool] = None,
            ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images, videos, audios
            )
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,
        cache_position: Optional[torch.Tensor] = None,
        videos: Union[Optional[torch.Tensor], Optional[List[torch.Tensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        outputs = self._get_output(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=True,
            cache_position=cache_position,
            videos=videos,
            return_dict=return_dict,
        )
        loss = outputs.loss
        
        if not return_dict:
            output = (outputs.logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    @torch.no_grad()
    def _get_generation(
        self, 
        input_ids: Optional[torch.Tensor] = None,
        videos: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 200,
        top_p: Optional[float] = 10.0,
        temperature: Optional[float] = 0.1,
        stopping_criteria: Optional[Callable] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs):
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if videos is not None:
            print('images is not none')
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, None, None, videos
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)

            batch_size, seq_length = attention_mask.shape
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).expand((batch_size, seq_length))
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        outputs = super().generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            # do_sample=True,
            # use_cache=True,
            stopping_criteria=stopping_criteria,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=True,
            output_attentions=True,
            **kwargs
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        videos: Optional[torch.Tensor] = None,
        stopping_criteria: Optional[Callable] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        print("kwargs: ", kwargs)
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        output_attentions = kwargs.pop("output_attentions", True)
        output_hidden_states = kwargs.pop("output_hidden_states", True)
        max_new_tokens = kwargs.pop("max_new_tokens", 200)
        top_p = kwargs.pop("top_p", 10.0)
        temperature = kwargs.pop("temperature", 0.1)
        # stopping_criteria = kwargs.pop("stopping_criteria", None)

        outputs = self._get_generation(
            input_ids=input_ids,
            videos=videos,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            temperature=temperature,
            stopping_criteria=stopping_criteria,
            position_ids=position_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs
        )

        
        return outputs  

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        videos = kwargs.pop("videos", None)

        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )

        if videos is not None:
            inputs["videos"] = videos
        return inputs

    def print_model_parameters(self, use_4bit=False):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        lora = 0
        image = 0
        video = 0
        audio = 0
        linear = 0
        llama = 0
        imagebind = 0
        for name, param in self.model.named_parameters():
            # print(f"{name}: {param.numel():,d} :: {param.requires_grad}")
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            if 'lora' in name:
                lora += num_params
            elif 'mm_input_projector' in name:
                linear += num_params
            elif name.startswith("layers") or name.startswith("embed_tokens") or name.startswith("norm.weight"):
                llama += num_params
            elif 'multimodal_tower' in name:
                imagebind += num_params
            else:
                pass

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        if use_4bit:
            trainable_params /= 2
        print(
            f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
        )
        print(f'linear params: {linear:,d} || imagebind params: {imagebind:,d} || llama params: {llama:,d} || lora params: {lora:,d}')


AutoConfig.register("motionepic_llama", MotionEpicConfig)
AutoModelForCausalLM.register(MotionEpicConfig, MotionEpicLlamaForCausalLM)

