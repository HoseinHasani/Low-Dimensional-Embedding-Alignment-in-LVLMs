import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.generation.logits_process import (
    LogitsProcessorList,
)
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
import transformers
from transformers.generation.utils import SampleOutput, SampleEncoderDecoderOutput, SampleDecoderOnlyOutput


from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from transformers import AutoTokenizer

import os
import matplotlib.pyplot as plt 

def sample_with_ensemble(
    self,
    input_ids: torch.LongTensor,
    sentence_evaluator: Callable,
    ensemble_size: int = 4,
    pseudo_sentence_length: int = 20,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    logits_warper: Optional[LogitsProcessorList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = True,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:

    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device)
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    final_input_ids = input_ids.clone()
    model_kwargs_main = model_kwargs.copy()
    finished = False

    while not finished:
        candidate_outputs = []
        candidate_attentions = []
        candidate_scores = []

        # generate ensemble_size pseudo-sentences
        for _ in range(ensemble_size):
            model_kwargs_tmp = copy.deepcopy(model_kwargs_main)
            input_tmp = final_input_ids.clone()
            decoder_attn_tmp = []
            token_count = 0
            eos_hit = False

            while token_count < pseudo_sentence_length and not eos_hit:
                model_inputs = self.prepare_inputs_for_generation(input_tmp, **model_kwargs_tmp)
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=True,
                )

                next_token_logits = outputs.logits[:, -1, :]
                next_token_scores = logits_processor(input_tmp, next_token_logits)
                next_token_scores = logits_warper(input_tmp, next_token_scores)
                probs = F.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

                input_tmp = torch.cat([input_tmp, next_tokens[:, None]], dim=-1)

                # collect attentions
                if self.config.is_encoder_decoder:
                    decoder_attn_tmp.append(outputs.cross_attentions)
                else:
                    decoder_attn_tmp.append(outputs.attentions)

                model_kwargs_tmp = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs_tmp, is_encoder_decoder=self.config.is_encoder_decoder
                )

                if any(next_tokens[0].item() == e for e in eos_token_id):
                    eos_hit = True
                token_count += 1

            candidate_outputs.append(input_tmp)
            candidate_attentions.append(decoder_attn_tmp)

        # evaluate ensemble
        for i in range(ensemble_size):
            score = sentence_evaluator(candidate_outputs[i], candidate_attentions[i])
            candidate_scores.append(score)

        best_idx = int(torch.tensor(candidate_scores).argmax())
        best_tokens = candidate_outputs[best_idx][:, final_input_ids.shape[1]:]
        final_input_ids = candidate_outputs[best_idx]

        model_kwargs_main = self._update_model_kwargs_for_generation(
            outputs, model_kwargs_main, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # stop if EOS seen or stopping criteria met
        if any(t.item() in eos_token_id for t in best_tokens[0]) or stopping_criteria(final_input_ids, None):
            finished = True

        if streamer is not None:
            streamer.put(best_tokens.cpu())

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        return SampleDecoderOnlyOutput(
            sequences=final_input_ids,
            attentions=None,
            hidden_states=None,
        )
    else:
        return final_input_ids




