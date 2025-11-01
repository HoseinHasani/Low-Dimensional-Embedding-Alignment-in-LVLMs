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


def sentence_evaluator(
    candidate_output_ids,
    candidate_attentions,
    classifier,
    image_token_index=IMAGE_TOKEN_INDEX,
    image_tokens_count=576,
    topk=20,
    layer_start=0,
    layer_end=32,
):
    try:
        image_idx = candidate_output_ids[0].tolist().index(image_token_index)
    except ValueError:
        return -float("inf")

    all_indices = list(range(image_idx + image_tokens_count, candidate_output_ids.shape[1]))
    all_attn = extract_top_attentions_by_steps(
        candidate_attentions,
        all_indices,
        image_idx,
        image_tokens_count,
        topk=topk,
        layer_start=layer_start,
        layer_end=layer_end,
    )
    preds = classifier.predict(all_attn)
    p_faithful = preds[:, 1] if preds.ndim > 1 and preds.shape[1] == 2 else 1 - preds
    return float(np.mean(p_faithful))



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
    eos_set = set(eos_token_id)

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # final sequence and model state
    final_input_ids = input_ids.clone()
    model_kwargs_main = copy.deepcopy(model_kwargs)

    finished = False

    # We do not collect global scores/attentions across the whole response in this version,
    # but we keep per-candidate per-token decoder attentions for the evaluator.
    while not finished:
        candidate_outputs = []
        candidate_attentions = []
        candidate_model_kwargs = []

        # sequentially sample ensemble candidates (each up to pseudo_sentence_length tokens)
        for _ in range(ensemble_size):
            model_kwargs_tmp = copy.deepcopy(model_kwargs_main)
            input_tmp = final_input_ids.clone()
            decoder_attn_tmp = []
            token_count = 0
            eos_hit = False

            # generate one pseudo-sentence for this candidate
            while token_count < pseudo_sentence_length and not eos_hit:
                model_inputs = self.prepare_inputs_for_generation(input_tmp, **model_kwargs_tmp)
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=True,
                    output_hidden_states=False,
                )

                next_token_logits = outputs.logits[:, -1, :]
                next_token_scores = logits_processor(input_tmp, next_token_logits)
                next_token_scores = logits_warper(input_tmp, next_token_scores)
                probs = F.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

                input_tmp = torch.cat([input_tmp, next_tokens[:, None]], dim=-1)

                # collect per-step attentions (store per-token attention objects)
                if self.config.is_encoder_decoder:
                    decoder_attn_tmp.append(outputs.cross_attentions)
                else:
                    decoder_attn_tmp.append(outputs.attentions)

                # update the temporary model kwargs to reflect autoregressive state for this candidate
                model_kwargs_tmp = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs_tmp, is_encoder_decoder=self.config.is_encoder_decoder
                )

                # check EOS in this token (works for batch size >=1; we treat candidate per-batch identically)
                # NOTE: this code assumes batch dimension is present; we treat token-wise EOS presence as ending candidate.
                # if any token in next_tokens is an EOS, mark eos_hit (typical case: batch size 1).
                if any(int(t.item()) in eos_set for t in next_tokens.view(-1)):
                    eos_hit = True

                token_count += 1

            candidate_outputs.append(input_tmp)
            candidate_attentions.append(decoder_attn_tmp)
            candidate_model_kwargs.append(model_kwargs_tmp)

        # evaluate candidates and pick best (assumes higher score == better / less hallucinated)
        candidate_scores = []
        for i in range(ensemble_size):
            # pass the incremental tokens (full sequence is okay; evaluator can use suffix if desired)
            score = sentence_evaluator(candidate_outputs[i], candidate_attentions[i])
            candidate_scores.append(float(score))

        best_idx = int(torch.tensor(candidate_scores).argmax().item())

        # append the chosen candidate's new tokens to final_input_ids
        # compute slice of newly produced tokens
        start_pos = final_input_ids.shape[1]
        best_full = candidate_outputs[best_idx]
        best_new_tokens = best_full[:, start_pos:]  # may be shorter than pseudo_sentence_length if EOS hit
        final_input_ids = best_full  # adopt the chosen candidate's full sequence
        model_kwargs_main = candidate_model_kwargs[best_idx]  # adopt the chosen candidate's model state

        # stream tokens if requested
        if streamer is not None:
            streamer.put(best_new_tokens.cpu())

        # stop if chosen candidate contained EOS or stopping criteria met
        # check EOS presence in newly appended tokens
        eos_in_best = any(int(t.item()) in eos_set for t in best_new_tokens.view(-1))
        if eos_in_best:
            finished = True

        # call stopping_criteria with scores placeholder (empty tuple) to match expected signature
        if stopping_criteria(final_input_ids, ()):
            finished = True

    if streamer is not None:
        streamer.end()

    # return with correct output class depending on model type
    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            # we don't currently collect encoder_attentions/hidden_states or decoder_hidden_states here;
            # return None for those fields (caller can adapt to collect if needed).
            return SampleEncoderDecoderOutput(
                sequences=final_input_ids,
                scores=None,
                encoder_attentions=None,
                encoder_hidden_states=None,
                decoder_attentions=None,
                cross_attentions=None,
                decoder_hidden_states=None,
            )
        else:
            return SampleDecoderOnlyOutput(
                sequences=final_input_ids,
                scores=None,
                attentions=None,
                hidden_states=None,
            )
    else:
        return final_input_ids





