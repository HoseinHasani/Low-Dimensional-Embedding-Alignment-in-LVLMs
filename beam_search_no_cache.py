import copy
import inspect
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import os
import sys
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifier.mlp_deployment import FPAttentionClassifier


from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from transformers import AutoTokenizer

import os
import matplotlib.pyplot as plt 
from llava.constants import IMAGE_TOKEN_INDEX
import numpy as np


def extract_top_attentions_by_steps(all_attentions, token_indices, image_token_index, image_tokens_count,
                                    topk=5, layer_start=0, layer_end=14, aggregate=True):
    """
    Extract top-k attended image and text tokens for each TP/FP token's subtoken steps.

    Args:
        all_attentions: tuple of length n_generated, each element is a list of layer attentions
                        [layer0, layer1, ...], each [batch, heads, seq, seq]
        token_indices: list of lists of subtoken indices for TP/FP tokens
        image_token_index: starting index of image tokens in the sequence
        image_tokens_count: number of image tokens in the sequence
        topk: number of top tokens to return
        layer_start, layer_end: layers to average (inclusive)
        aggregate: if True, averages over subtokens at the end;
                   if False, keeps each subtoken result separately.

    Returns:
        dict with keys:
            {
                "image": [{"token_indices": [...],
                           "subtoken_results": [{"idx": int, "topk_indices": [...], "topk_values": [...]}],
                           "mean_topk_values": [...], "mean_topk_indices": [...]}],
                "text":  [ ... same structure ... ]
            }
    """

    results = {}
    batch_size = all_attentions[0][0].shape[0]
    assert batch_size == 1, "Only batch size 1 supported."

    n_layers_total = len(all_attentions[0])
    layer_end = min(layer_end, n_layers_total - 1)
    
    token_indices = np.ravel(token_indices)

    for idx in token_indices:

        # (num_layers, B, H, S, S)
        attn_matrix = all_attentions[idx][layer_start:layer_end + 1]
        target_attn_vec = [attn_matrix[i][0].cpu().numpy() for i in range(len(attn_matrix)) ]  
        target_attn_vec = np.array(target_attn_vec).squeeze()

        image_attn = target_attn_vec[..., image_token_index:image_token_index + image_tokens_count]

        topk_indices = np.argsort(image_attn, axis=-1)[..., -topk:][..., ::-1]

        topk_values = np.take_along_axis(image_attn, topk_indices, axis=-1)

        results[idx] = topk_values

    return results



def sentence_evaluator(
    candidate_attentions,
    classifier,
    token_offset,
    image_token_index=35,
    image_tokens_count=576,
    topk=20,
    layer_start=0,
    layer_end=32,
    threshold=0.5
):

    all_indices = [[i] for i in range(len(candidate_attentions))]
    all_attn = extract_top_attentions_by_steps(
        candidate_attentions,
        all_indices,
        image_token_index,
        image_tokens_count,
        topk=topk,
        layer_start=layer_start,
        layer_end=layer_end,
    )

    shifted_attn = {k + token_offset: v for k,v in all_attn.items()}
    preds = classifier.predict(shifted_attn)
    n_fp = np.sum([preds[k] > threshold for k in preds.keys()])
    return -n_fp





def _prepare_prompt_state(self, prompt_input_ids, model_kwargs):
    """
    Run a single forward on the prompt to compute encoder outputs or any
    other fixed embeddings we want to reuse. Return a shallow-cloned model_kwargs
    that contains frozen prompt state but DOES NOT include decoder past_key_values.
    """
    mk = copy.deepcopy(model_kwargs)
    # Remove any past_key_values so forward computes from scratch
    mk.pop("past_key_values", None)
    # Run a forward once to get encoder_outputs or other persistent states (encoder-decoder)
    model_inputs = self.prepare_inputs_for_generation(prompt_input_ids, **mk)
    # For encoder-decoder models, this produces encoder_outputs that we can reuse.
    # For decoder-only models, there is no encoder_outputs to cache; we will just
    # recompute full forward during candidate generation.
    outputs = self(**model_inputs, return_dict=True)
    # Copy back encoder-related outputs into mk in a safe detached manner, if present
    if hasattr(outputs, 'encoder_last_hidden_state') or hasattr(outputs, 'encoder_outputs'):
        # Different HF models name things differently; use common 'encoder_outputs' if present
        encoder_outputs = getattr(outputs, "encoder_outputs", None)
        if encoder_outputs is None and hasattr(outputs, "encoder_last_hidden_state"):
            # wrap into a simple object mimic if needed
            encoder_outputs = outputs.encoder_last_hidden_state
        mk["encoder_outputs"] = encoder_outputs
    # detach any tensors to avoid autograd history
    for k, v in list(mk.items()):
        if isinstance(v, torch.Tensor):
            mk[k] = v.detach()
    return mk


def _generate_segment_from_prefix(self, prefix_input_ids, model_kwargs_prompt, length, logits_processor, logits_warper, eos_token_id, output_attentions=True, temperature=1.0, top_p=None):
    """
    Generate up to `length` tokens from the model starting from `prefix_input_ids`.
    This function NEVER reuses past_key_values between candidates — it always runs
    step-by-step from the full prefix (but it can reuse encoder outputs included in model_kwargs_prompt).
    Returns:
      - generated_ids (tensor): shape (1, prefix_len + gen_len)
      - per_step_attentions: list of length gen_len where each element is outputs.cross_attentions (or outputs.attentions)
      - stopped_early: whether EOS was generated in this segment
    """
    device = prefix_input_ids.device
    gen_ids = prefix_input_ids.clone()
    per_step_attns = []
    stopped_early = False

    for step in range(length):
        # make a fresh shallow copy to avoid mutation across candidates
        mk = copy.deepcopy(model_kwargs_prompt)
        # drop stale masks and rebuild them
        mk.pop("attention_mask", None)
        mk["attention_mask"] = torch.ones(gen_ids.shape, dtype=torch.long, device=gen_ids.device)
    
        model_inputs = self.prepare_inputs_for_generation(gen_ids, **mk)
        outputs = self(**model_inputs, return_dict=True, output_attentions=output_attentions)

        # logits for the next token
        next_token_logits = outputs.logits[:, -1, :]

        # temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # apply processors/warpers
        next_token_scores = logits_processor(gen_ids, next_token_logits)
        next_token_scores = logits_warper(gen_ids, next_token_scores)

        probs = F.softmax(next_token_scores, dim=-1)

        # optional nucleus sampling
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            # mask out tokens beyond top_p
            mask = cumsum > top_p
            # ensure at least one token remains
            mask[..., 0] = False
            # build filtered logits
            filtered_logits = next_token_scores.clone()
            # mask positions in original coords
            filtered_logits[mask.scatter(-1, sorted_idx, mask)] = -float("inf")
            probs = F.softmax(filtered_logits, dim=-1)

        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)  # shape (B,)
        gen_ids = torch.cat([gen_ids, next_tokens[:, None]], dim=-1)

        # store attentions for this step
        per_step_attns.append(outputs.cross_attentions if self.config.is_encoder_decoder else outputs.attentions)

        # check eos
        if int(next_tokens[0].item()) in eos_token_id:
            stopped_early = True
            break

    return gen_ids, per_step_attns, stopped_early

# ----------------- main function: recompute-after-selection -----------------

def sample_with_ensemble(
    self,
    input_ids: torch.LongTensor,
    ensemble_size: int = 5,
    pseudo_sentence_length: int = 20,
    search_start: int = 2,
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
    classifier_kwargs: dict = None,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    **model_kwargs,
):
    """
    Recompute-after-selection ensemble sampling.
    - cache prompt-level state (encoder outputs) once
    - for each ensemble step, generate candidates from scratch from prompt + recorded_response
    - pick best candidate and commit its tokens to recorded_response
    """

    # ---------- Classifier ----------
    classifier = FPAttentionClassifier(
        model_path=classifier_kwargs.get("model_path") if classifier_kwargs else "/home/rz15dire/Ensemble/experiments/eval/classifier_outputs/llava_temp1/pytorch_mlp_exp__ent1_gin0/model/pytorch_mlp_with_l2.pt",
        scaler_path=classifier_kwargs.get("scaler_path") if classifier_kwargs else "/home/rz15dire/Ensemble/experiments/eval/classifier_outputs/llava_temp1/pytorch_mlp_exp__ent1_gin0/model/scaler.pt",
        n_layers=classifier_kwargs.get("n_layers", 32) if classifier_kwargs else 32,
        n_heads=classifier_kwargs.get("n_heads", 32) if classifier_kwargs else 32,
        use_entropy=classifier_kwargs.get("use_entropy", True) if classifier_kwargs else True,
        use_gini=classifier_kwargs.get("use_gini", False) if classifier_kwargs else False,
    )

    # ---------- setup ----------
    logits_processor = logits_processor or LogitsProcessorList()
    logits_warper = logits_warper or LogitsProcessorList()
    stopping_criteria = stopping_criteria or StoppingCriteriaList()
    if max_length is not None:
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

    pad_token_id = pad_token_id or self.generation_config.pad_token_id
    eos_token_id = eos_token_id or self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_set = set(eos_token_id)
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device)

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    device = input_ids.device

    # ---------- cache prompt-level state once ----------
    # model_kwargs may contain encoder inputs, attention masks etc.
    model_kwargs_prompt = _prepare_prompt_state(self, input_ids, model_kwargs)
    # ensure no past_key_values retained
    model_kwargs_prompt.pop("past_key_values", None)

    # recorded_response = tokens selected (after the prompt)
    recorded_response = []  # list of ints
    final_input_ids = input_ids.clone()
    finished = False
    total_generated = 0

    # ---------- Warm-up: produce `search_start` tokens using normal sampling and commit them ----------
    while total_generated < search_start and not finished:
        # generate one token step using the same recompute logic (prefix = prompt + recorded_response)
        prefix_ids = torch.cat([input_ids, torch.tensor(recorded_response, device=device).unsqueeze(0)] , dim=-1) if len(recorded_response) > 0 else input_ids
        gen_ids, attns, stopped = _generate_segment_from_prefix(
            self,
            prefix_ids,
            model_kwargs_prompt,
            length=1,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            eos_token_id=eos_token_id,
            output_attentions=output_attentions,
            temperature=temperature,
            top_p=top_p,
        )
        # gen_ids is prefix + new token
        new_token = gen_ids[0, -1].item()
        recorded_response.append(new_token)
        final_input_ids = torch.cat([input_ids, torch.tensor(recorded_response, device=device).unsqueeze(0)], dim=-1)
        total_generated += 1
        if new_token in eos_set:
            finished = True
        if streamer is not None:
            streamer.put(torch.tensor([new_token]).cpu())

    # ---------- Ensemble-guided sampling (recompute after selection) ----------
    while not finished:
        candidate_full_ids = []
        candidate_attns = []
        candidate_stopped = []

        # For each ensemble member, regenerate a pseudo-sentence from scratch
        for _ in range(ensemble_size):
            # rebuild prefix: prompt + recorded_response
            prefix_ids = torch.cat([input_ids, torch.tensor(recorded_response, device=device).unsqueeze(0)] , dim=-1) if len(recorded_response) > 0 else input_ids

            # generate a segment from scratch (no decoder past reuse across candidates)
            gen_ids, per_step_attns, stopped = _generate_segment_from_prefix(
                self,
                prefix_ids,
                model_kwargs_prompt,
                length=pseudo_sentence_length,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                eos_token_id=eos_token_id,
                output_attentions=output_attentions,
                temperature=temperature,
                top_p=top_p,
            )

            candidate_full_ids.append(gen_ids)         # full: prefix + generated segment
            candidate_attns.append(per_step_attns)     # list of per-step attentions (len <= pseudo_sentence_length)
            candidate_stopped.append(stopped)

        # ---------- Evaluate candidates ----------
        candidate_scores = []
        # token_offset: number of already committed tokens after the prompt
        token_offset = len(recorded_response)
        for i in range(ensemble_size):
            # candidate_attns[i] is a list of length gen_len where each entry is cross_attns / attentions
            score = sentence_evaluator(candidate_attns[i], classifier, token_offset)
            candidate_scores.append(float(score))

        best_idx = int(np.argmax(candidate_scores))

        # ---------- Commit best candidate tokens to recorded_response ----------
        best_full = candidate_full_ids[best_idx]  # (1, prefix_len + gen_len)
        # compute newly produced tokens (slice after prefix)
        prefix_len = input_ids.shape[1] + len(recorded_response)
        new_tokens = best_full[:, prefix_len:].cpu().numpy().tolist()[0]  # list of ints
        # If candidate stopped early, we already included EOS in new_tokens
        recorded_response.extend(new_tokens)
        final_input_ids = torch.cat([input_ids, torch.tensor(recorded_response, device=device).unsqueeze(0)], dim=-1)
        total_generated += len(new_tokens)

        # stream the appended new tokens
        if streamer is not None and len(new_tokens) > 0:
            streamer.put(torch.tensor(new_tokens).unsqueeze(0).cpu())

        # ---------- stopping checks ----------
        if any(t in eos_set for t in new_tokens):
            finished = True
            break
        if stopping_criteria(final_input_ids, ()):
            finished = True
            break

        # Safety / max tokens
        if max_length is not None and final_input_ids.shape[1] >= max_length:
            finished = True
            break

    if streamer is not None:
        streamer.end()

    # ---------- Return ----------
    if return_dict_in_generate:
        # for encoder-decoder, return SampleEncoderDecoderOutput skeleton
        if self.config.is_encoder_decoder:
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
    



def evolve_beam_search():
    transformers.generation.utils.GenerationMixin.sample = sample_with_ensemble
    # sample is now a protected function in the latest Transformers library
    transformers.generation.utils.GenerationMixin._sample = sample_with_ensemble


