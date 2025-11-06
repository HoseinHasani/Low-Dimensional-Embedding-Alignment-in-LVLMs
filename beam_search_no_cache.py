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
import gc

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


# def extract_top_attentions_by_steps(all_attentions, token_indices, image_token_index, image_tokens_count,
#                                     topk=5, layer_start=0, layer_end=14, aggregate=True):
#     """
#     Extract top-k attended image and text tokens for each TP/FP token's subtoken steps.

#     Args:
#         all_attentions: tuple of length n_generated, each element is a list of layer attentions
#                         [layer0, layer1, ...], each [batch, heads, seq, seq]
#         token_indices: list of lists of subtoken indices for TP/FP tokens
#         image_token_index: starting index of image tokens in the sequence
#         image_tokens_count: number of image tokens in the sequence
#         topk: number of top tokens to return
#         layer_start, layer_end: layers to average (inclusive)
#         aggregate: if True, averages over subtokens at the end;
#                    if False, keeps each subtoken result separately.

#     Returns:
#         dict with keys:
#             {
#                 "image": [{"token_indices": [...],
#                            "subtoken_results": [{"idx": int, "topk_indices": [...], "topk_values": [...]}],
#                            "mean_topk_values": [...], "mean_topk_indices": [...]}],
#                 "text":  [ ... same structure ... ]
#             }
#     """

#     results = {}
#     batch_size = all_attentions[0][0].shape[0]
#     assert batch_size == 1, "Only batch size 1 supported."

#     n_layers_total = len(all_attentions[0])
#     layer_end = min(layer_end, n_layers_total - 1)
    
#     token_indices = np.ravel(token_indices)

#     for idx in token_indices:

#         # (num_layers, B, H, S, S)
#         attn_matrix = all_attentions[idx][layer_start:layer_end + 1]
#         target_attn_vec = [attn_matrix[i][0].cpu().numpy() for i in range(len(attn_matrix)) ]  
#         target_attn_vec = np.array(target_attn_vec).squeeze()

#         image_attn = target_attn_vec[..., image_token_index:image_token_index + image_tokens_count]

#         topk_indices = np.argsort(image_attn, axis=-1)[..., -topk:][..., ::-1]

#         topk_values = np.take_along_axis(image_attn, topk_indices, axis=-1)

#         results[idx] = topk_values

#     return results



# def sentence_evaluator(
#     candidate_attentions,
#     classifier,
#     token_offset,
#     image_token_index=35,
#     image_tokens_count=576,
#     topk=20,
#     layer_start=0,
#     layer_end=32,
#     threshold=0.5
# ):

#     all_indices = [[i] for i in range(len(candidate_attentions))]
#     all_attn = extract_top_attentions_by_steps(
#         candidate_attentions,
#         all_indices,
#         image_token_index,
#         image_tokens_count,
#         topk=topk,
#         layer_start=layer_start,
#         layer_end=layer_end,
#     )

#     shifted_attn = {k + token_offset: v for k,v in all_attn.items()}
#     preds = classifier.predict(shifted_attn)
#     n_fp = np.sum([preds[k] > threshold for k in preds.keys()])
#     return -n_fp




# --------------------- small helpers ---------------------

def _prepare_prompt_state(self, prompt_input_ids, model_kwargs):
    """
    Run a single forward on the prompt to compute encoder outputs (if any).
    Return model_kwargs_prompt which includes detached encoder outputs (if present)
    and does NOT include decoder past_key_values.
    """
    mk = copy.deepcopy(model_kwargs)
    mk.pop("past_key_values", None)
    mk.pop("attention_mask", None)  # we'll rebuild masks when generating
    with torch.no_grad():
        model_inputs = self.prepare_inputs_for_generation(prompt_input_ids, **mk)
        outputs = self(**model_inputs, return_dict=True)
    # stash encoder outputs if present
    encoder_outputs = getattr(outputs, "encoder_outputs", None)
    if encoder_outputs is None and hasattr(outputs, "encoder_last_hidden_state"):
        encoder_outputs = outputs.encoder_last_hidden_state
    if encoder_outputs is not None:
        # detach to avoid autograd and keep on same device
        mk["encoder_outputs"] = encoder_outputs.detach()
    # ensure no past_key_values
    mk.pop("past_key_values", None)
    return mk


def _extract_compact_topk_from_step(step_attns, image_token_index, image_tokens_count, topk=20, layer_start=0, layer_end=None):
    """
    Given step_attns = outputs.cross_attentions OR outputs.attentions for a single step,
    produce a compact top-k image-attention vector (numpy array) to send to classifier.

    step_attns: tuple/list over layers; each layer tensor has shape (B, H, S, S)
                (B is batch, typically 1)
    Returns: np.array of shape (topk,) containing topk image-attention values (descending)
    """
    if layer_end is None:
        layer_end = len(step_attns) - 1
    # Collect image-attention vectors per layer/head for the LAST token (query index = S-1)
    # We'll average across heads and layers then pick topk image token values.
    # This keeps small memory footprint (we don't store full SxS).
    B = step_attns[0].shape[0]
    assert B == 1, "Only batch size 1 supported for classifier extraction."
    layer_values = []
    for li, layer_tensor in enumerate(step_attns[layer_start: layer_end + 1]):
        # layer_tensor shape: (B, H, S, S)
        lt = layer_tensor[0]  # (H, S, S)
        S = lt.shape[-1]
        query_row = lt[:, S - 1, :]  # (H, S)
        # get image slice
        img_slice = query_row[:, image_token_index:image_token_index + image_tokens_count]  # (H, image_tokens)
        # average across heads -> vector (image_tokens,)
        head_mean = img_slice.mean(dim=0).cpu().numpy()  # numpy float32
        layer_values.append(head_mean)
    # average across selected layers
    if len(layer_values) == 0:
        # fallback: zeros
        combined = np.zeros((image_tokens_count,), dtype=np.float32)
    else:
        combined = np.stack(layer_values, axis=0).mean(axis=0)  # (image_tokens,)
    # get topk values (descending)
    if topk >= combined.shape[0]:
        topk_vals = np.sort(combined)[::-1]
    else:
        topk_idx = np.argpartition(combined, -topk)[-topk:]
        topk_vals = np.sort(combined[topk_idx])[::-1]
    return topk_vals.astype(np.float32)


def _score_candidate_with_compact_attns(compact_attn_list, classifier, token_offset, topk=20, threshold=0.5):
    """
    compact_attn_list: list of per-step numpy arrays (topk values) in generation order for the candidate
    Build the dict mapping absolute token idx -> topk_values and call classifier.predict.
    Return -n_fp (matching your previous sentence_evaluator return).
    """
    # Build dict mapping absolute token index -> topk_values
    all_feats = {}
    for step_idx, topk_vals in enumerate(compact_attn_list):
        abs_idx = token_offset + step_idx
        all_feats[abs_idx] = topk_vals  # numpy array
    preds = classifier.predict(all_feats)  # expect dict abs_idx -> score
    n_fp = np.sum([preds[k] > threshold for k in preds.keys()])
    return -int(n_fp)


# ----------------- generation helper (memory safe) -----------------

def _generate_segment_from_prefix_memory_safe(
    self,
    prefix_input_ids,
    model_kwargs_prompt,
    length,
    logits_processor,
    logits_warper,
    eos_token_id,
    output_attentions=True,
    image_token_index=35,
    image_tokens_count=576,
    topk_for_classifier=20,
    layer_start=0,
    layer_end=None,
    temperature=1.0,
    top_p=None,
):
    """
    Memory-safe generation of a segment from prefix_input_ids.
    Returns:
      - gen_ids (tensor) shape (1, prefix_len + gen_len)
      - compact_attn_list: list (len=gen_len) of numpy arrays (topk values) (moved to CPU)
      - stopped_early (bool)
    Notes:
      - This runs with torch.no_grad(), and any attention tensors are processed immediately
        into small numpy topk vectors and then freed.
    """
    device = prefix_input_ids.device
    gen_ids = prefix_input_ids.clone()
    compact_attns = []
    stopped_early = False

    for step in range(length):
        # copy prompt model kwargs but drop stale attention masks
        mk = copy.deepcopy(model_kwargs_prompt)
        mk.pop("attention_mask", None)
        # create fresh attention mask matching gen_ids
        mk["attention_mask"] = torch.ones(gen_ids.shape, dtype=torch.long, device=device)

        # Run one forward (no grad)
        with torch.no_grad():
            model_inputs = self.prepare_inputs_for_generation(gen_ids, **mk)
            outputs = self(**model_inputs, return_dict=True, output_attentions=output_attentions)

            # compute next token logits
            next_token_logits = outputs.logits[:, -1, :]

            # temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            next_token_scores = logits_processor(gen_ids, next_token_logits)
            next_token_scores = logits_warper(gen_ids, next_token_scores)
            probs = F.softmax(next_token_scores, dim=-1)

            # nucleus filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > top_p
                mask[..., 0] = False
                # scatter mask back
                filtered_logits = next_token_scores.clone()
                filtered_logits[mask.scatter(-1, sorted_idx, mask)] = -float("inf")
                probs = F.softmax(filtered_logits, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            gen_ids = torch.cat([gen_ids, next_tokens[:, None]], dim=-1)

            # if we requested attentions, immediately compute compact topk for this step
            if output_attentions:
                step_attns = outputs.cross_attentions if self.config.is_encoder_decoder else outputs.attentions
                # step_attns: tuple/list per layer of tensors (B,H,S,S)
                topk_vals = _extract_compact_topk_from_step(
                    step_attns,
                    image_token_index=image_token_index,
                    image_tokens_count=image_tokens_count,
                    topk=topk_for_classifier,
                    layer_start=layer_start,
                    layer_end=layer_end,
                )  # numpy array on CPU
                compact_attns.append(topk_vals)  # stored on CPU, small
            # check eos
            if int(next_tokens[0].item()) in eos_token_id:
                stopped_early = True

            # Immediately free big tensors that outputs held
            # detach & delete outputs to free GPU memory
            del outputs
            torch.cuda.empty_cache()

        # break early on EOS
        if stopped_early:
            break

    return gen_ids, compact_attns, stopped_early


# ---------------- main: recompute-after-selection, memory-safe ----------------

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
    image_token_index: int = 35,
    image_tokens_count: int = 576,
    topk_for_classifier: int = 20,
    layer_start: int = 0,
    layer_end: Optional[int] = None,
    **model_kwargs,
):
    """
    Recompute-after-selection ensemble sampling, memory-safe:
      - compute encoder/prompt state once
      - for each candidate: recompute its segment, immediately extract small CPU-side features
      - score candidates using classifier on CPU features
      - commit best candidate tokens (only indices needed)
    """

    # ---------- classifier ----------
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

    device = input_ids.device
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # ---------- cache prompt-level state once ----------
    model_kwargs_prompt = _prepare_prompt_state(self, input_ids, model_kwargs)
    model_kwargs_prompt.pop("past_key_values", None)

    recorded_response = []  # list of ints (committed tokens after prompt)
    final_input_ids = input_ids.clone()
    finished = False
    total_generated = 0

    # ---------- warm-up (generate search_start tokens) ----------
    while total_generated < search_start and not finished:
        prefix_ids = torch.cat([input_ids, torch.tensor(recorded_response, device=device).unsqueeze(0)], dim=-1) if len(recorded_response) > 0 else input_ids
        gen_ids, compact_attns, stopped = _generate_segment_from_prefix_memory_safe(
            self,
            prefix_ids,
            model_kwargs_prompt,
            length=1,
            logits_processor=logits_processor,
            logits_warper=logits_warper,
            eos_token_id=eos_token_id,
            output_attentions=output_attentions,
            image_token_index=image_token_index,
            image_tokens_count=image_tokens_count,
            topk_for_classifier=topk_for_classifier,
            layer_start=layer_start,
            layer_end=layer_end,
            temperature=temperature,
            top_p=top_p,
        )
        new_token = int(gen_ids[0, -1].item())
        recorded_response.append(new_token)
        final_input_ids = torch.cat([input_ids, torch.tensor(recorded_response, device=device).unsqueeze(0)], dim=-1)
        total_generated += 1
        if new_token in eos_set:
            finished = True
        if streamer is not None:
            streamer.put(torch.tensor([new_token]).cpu())

    # ---------- ensemble-guided sampling ----------
    while not finished:
        candidate_full_ids = []
        candidate_compact_attns = []  # list of list of numpy arrays
        candidate_stopped = []

        # generate each candidate from scratch (memory freed between iterations)
        for _ in range(ensemble_size):
            prefix_ids = torch.cat([input_ids, torch.tensor(recorded_response, device=device).unsqueeze(0)], dim=-1) if len(recorded_response) > 0 else input_ids

            gen_ids, compact_attns, stopped = _generate_segment_from_prefix_memory_safe(
                self,
                prefix_ids,
                model_kwargs_prompt,
                length=pseudo_sentence_length,
                logits_processor=logits_processor,
                logits_warper=logits_warper,
                eos_token_id=eos_token_id,
                output_attentions=output_attentions,
                image_token_index=image_token_index,
                image_tokens_count=image_tokens_count,
                topk_for_classifier=topk_for_classifier,
                layer_start=layer_start,
                layer_end=layer_end,
                temperature=temperature,
                top_p=top_p,
            )

            candidate_full_ids.append(gen_ids.cpu())      # small tensor of ids on CPU
            candidate_compact_attns.append(compact_attns) # small list of numpy arrays on CPU
            candidate_stopped.append(stopped)

            # free any leftover memory and help python GC
            gc.collect()
            torch.cuda.empty_cache()

        # ---------- evaluate candidates (classifier on CPU features) ----------
        token_offset = len(recorded_response)
        candidate_scores = []
        for i in range(len(candidate_compact_attns)):
            score = _score_candidate_with_compact_attns(candidate_compact_attns[i], classifier, token_offset, topk=topk_for_classifier)
            candidate_scores.append(float(score))

        best_idx = int(np.argmax(candidate_scores))

        # ---------- commit chosen tokens (only indices) ----------
        best_full = candidate_full_ids[best_idx]   # cpu tensor (1, prefix_len + gen_len)
        prefix_len = input_ids.shape[1] + len(recorded_response)
        new_tokens = best_full[:, prefix_len:].cpu().numpy().tolist()[0]
        recorded_response.extend(new_tokens)
        final_input_ids = torch.cat([input_ids, torch.tensor(recorded_response, device=device).unsqueeze(0)], dim=-1)
        total_generated += len(new_tokens)

        # stream appended tokens
        if streamer is not None and len(new_tokens) > 0:
            streamer.put(torch.tensor(new_tokens).unsqueeze(0).cpu())

        # stopping
        if any(t in eos_set for t in new_tokens):
            finished = True
            break
        if stopping_criteria(final_input_ids, ()):
            finished = True
            break
        if max_length is not None and final_input_ids.shape[1] >= max_length:
            finished = True
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
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


