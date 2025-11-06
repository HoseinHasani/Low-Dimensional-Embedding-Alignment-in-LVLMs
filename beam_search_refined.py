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


# -------------------------- Helpers --------------------------



def infer_cached_seq_len_from_past(past_key_values):
    """
    Return the seq_len(s) found in past_key_values.
    past_key_values: tuple of length num_layers, each element is a tuple/tensors for that layer.
    Returns a list of detected seq_lens (one per layer key tensor); usually they should all be equal.
    """
    if past_key_values is None:
        return None
    seq_lens = []
    for layer in past_key_values:
        # layer is usually (k, v) or (k, v, ...) where k shape (B, H, seq_len, head_dim)
        # find the first tensor in layer that has 4 dims and use its seq dim.
        found = False
        for t in layer:
            if isinstance(t, torch.Tensor) and t.dim() >= 3:
                # Heuristic: for key/value shapes are (B,H,seq_len,head_dim), seq_len is dim 2
                # But in some implementations it's dim 2 or 3; we look for the largest spatial dim.
                # We'll assume seq_len is dim -2 if ndims==4
                if t.dim() == 4:
                    seq_lens.append(t.shape[2])
                    found = True
                    break
                elif t.dim() == 3:
                    seq_lens.append(t.shape[1])
                    found = True
                    break
        if not found:
            seq_lens.append(None)
    return seq_lens

def get_effective_cached_len(past_key_values):
    seq_lens = infer_cached_seq_len_from_past(past_key_values)
    if seq_lens is None:
        return None
    # return the most common non-None seq len
    seq_lens = [s for s in seq_lens if s is not None]
    if not seq_lens:
        return None
    # If layers disagree, return max (and warn)
    if len(set(seq_lens)) > 1:
        print(f"[WARN] different seq_lens across layers: {seq_lens}")
    return max(seq_lens)

def trim_past_key_values_to_length(past_key_values, target_len):
    """
    Trim past_key_values along the sequence axis to keep only the last target_len positions.
    Returns new past_key_values with tensors cloned/moved appropriately.
    """
    if past_key_values is None:
        return None
    new_past = []
    for layer in past_key_values:
        new_layer = []
        for t in layer:
            if not isinstance(t, torch.Tensor):
                new_layer.append(t)
                continue
            # Handle 4D (B,H,seq_len,head_dim) common case:
            if t.dim() == 4:
                seq_len = t.shape[2]
                if target_len > seq_len:
                    # Can't trim to longer than available
                    raise ValueError(f"target_len {target_len} > cached seq_len {seq_len}")
                # keep last `target_len` positions along dim 2:
                new_t = t[..., -target_len:, :].detach().clone()
                new_layer.append(new_t)
            elif t.dim() == 3:
                seq_len = t.shape[1]
                if target_len > seq_len:
                    raise ValueError(f"target_len {target_len} > cached seq_len {seq_len}")
                new_t = t[..., -target_len:].detach().clone()
                new_layer.append(new_t)
            else:
                # other shapes (e.g., 2D) just clone
                new_layer.append(t.detach().clone())
        new_past.append(tuple(new_layer))
    return tuple(new_past)

def safe_align_cache_to_input(model_kwargs, input_ids, strict=True):
    """
    Ensure model_kwargs['past_key_values'] matches len(input_ids).
    If mismatch:
      - if strict: raise
      - else: try to trim past to match (fast), or return None to indicate recompute needed
    Returns updated_model_kwargs.
    """
    pk = model_kwargs.get("past_key_values", None)
    if pk is None:
        return model_kwargs
    cached_len = get_effective_cached_len(pk)
    input_len = input_ids.shape[1]
    if cached_len == input_len:
        return model_kwargs
    if cached_len is None:
        return model_kwargs
    if cached_len > input_len:
        # trim
        try:
            model_kwargs["past_key_values"] = trim_past_key_values_to_length(pk, input_len)
            return model_kwargs
        except Exception as e:
            if strict:
                raise
            print(f"[WARN] could not trim past: {e}; caller should recompute from scratch")
            return model_kwargs
    else:
        # cached shorter than inputs -> can't fix by trimming, need to recompute or pad (not safe)
        if strict:
            raise ValueError(f"cached length ({cached_len}) shorter than input tokens ({input_len})")
        print("[WARN] cached shorter than input; caller should recompute from scratch")
        return model_kwargs
    
    


def sample_with_ensemble(
    self,
    input_ids: torch.LongTensor,
    ensemble_size: int = 4,
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
    **model_kwargs,
) -> Union[SampleOutput, torch.LongTensor]:
    """
    Ensemble-based sampling (research-stable version):
    - Uses recomputation of cache after each chosen pseudo-sentence (safe, slower).
    - Avoids cache mismatch errors (past_key_values always aligned with final_input_ids).
    - Uses external classifier-based evaluator for hallucination scoring.
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    classifier = FPAttentionClassifier(
        model_path="/home/rz15dire/Ensemble/experiments/eval/classifier_outputs/llava_temp1/"
                   "pytorch_mlp_exp__ent1_gin0/model/pytorch_mlp_with_l2.pt",
        scaler_path="/home/rz15dire/Ensemble/experiments/eval/classifier_outputs/llava_temp1/"
                    "pytorch_mlp_exp__ent1_gin0/model/scaler.pt",
        n_layers=32,
        n_heads=32,
        use_entropy=True,
        use_gini=False,
    )

    logits_processor = logits_processor or LogitsProcessorList()
    stopping_criteria = stopping_criteria or StoppingCriteriaList()
    if max_length is not None:
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    logits_warper = logits_warper or LogitsProcessorList()

    pad_token_id = pad_token_id or self.generation_config.pad_token_id
    eos_token_id = eos_token_id or self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_set = set(eos_token_id)

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    final_input_ids = input_ids.clone()
    model_kwargs_main = copy.deepcopy(model_kwargs)
    finished = False
    total_generated = 0

    # -------------------------------------------------------------------------
    # 1. Normal sampling before ensemble starts (warm-up)
    # -------------------------------------------------------------------------
    while total_generated < search_start and not finished:
        model_inputs = self.prepare_inputs_for_generation(final_input_ids, **model_kwargs_main)
        outputs = self(**model_inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]

        next_token_scores = logits_processor(final_input_ids, next_token_logits)
        next_token_scores = logits_warper(final_input_ids, next_token_scores)
        probs = F.softmax(next_token_scores, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

        final_input_ids = torch.cat([final_input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs_main = self._update_model_kwargs_for_generation(
            outputs, model_kwargs_main, is_encoder_decoder=self.config.is_encoder_decoder
        )

        total_generated += 1

        if int(next_tokens[0].item()) in eos_set:
            finished = True

        if streamer is not None:
            streamer.put(next_tokens.cpu())

    # -------------------------------------------------------------------------
    # 2. Ensemble sampling loop
    # -------------------------------------------------------------------------
    while not finished:
        candidate_outputs = []
        candidate_attentions = []
        candidate_scores = []
        candidate_model_kwargs = []

        # ---------------------------------------------------------
        # Generate pseudo-sentences for ensemble members
        # ---------------------------------------------------------
        for _ in range(ensemble_size):
            model_kwargs_tmp = copy.deepcopy(model_kwargs_main)
            input_tmp = final_input_ids.clone()
            decoder_attn_tmp = []
            token_count = 0
            eos_hit = False

            while token_count < pseudo_sentence_length and not eos_hit:
                model_inputs = self.prepare_inputs_for_generation(input_tmp, **model_kwargs_tmp)
                outputs = self(**model_inputs, return_dict=True, output_attentions=True)
                next_token_logits = outputs.logits[:, -1, :]

                next_token_scores = logits_processor(input_tmp, next_token_logits)
                next_token_scores = logits_warper(input_tmp, next_token_scores)
                probs = F.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

                input_tmp = torch.cat([input_tmp, next_tokens[:, None]], dim=-1)

                if self.config.is_encoder_decoder:
                    decoder_attn_tmp.append(outputs.cross_attentions)
                else:
                    decoder_attn_tmp.append(outputs.attentions)

                model_kwargs_tmp = self._update_model_kwargs_for_generation(
                    outputs, model_kwargs_tmp, is_encoder_decoder=self.config.is_encoder_decoder
                )

                if any(int(t.item()) in eos_set for t in next_tokens.view(-1)):
                    eos_hit = True

                token_count += 1

            candidate_outputs.append(input_tmp)
            candidate_attentions.append(decoder_attn_tmp)
            candidate_model_kwargs.append(model_kwargs_tmp)

        # ---------------------------------------------------------
        # Evaluate ensemble candidates
        # ---------------------------------------------------------
        for i in range(ensemble_size):
            score = sentence_evaluator(
                candidate_attentions[i],
                classifier,
                token_offset=total_generated,
            )
            candidate_scores.append(float(score))

        best_idx = int(torch.tensor(candidate_scores).argmax().item())
        best_full = candidate_outputs[best_idx]
        best_new_tokens = best_full[:, final_input_ids.shape[1]:]
        total_generated += best_new_tokens.shape[1]

        # ---------------------------------------------------------
        # Recompute from scratch for chosen candidate (cache-safe)
        # ---------------------------------------------------------
        final_input_ids = best_full

        # Recompute model state from scratch (drop old cache)
        model_inputs = self.prepare_inputs_for_generation(
            final_input_ids, past_key_values=None, **{k: v for k, v in model_kwargs_main.items() if k != "past_key_values"}
        )
        outputs = self(**model_inputs, return_dict=True)
        model_kwargs_main = self._update_model_kwargs_for_generation(
            outputs, {}, is_encoder_decoder=self.config.is_encoder_decoder
        )

        if streamer is not None:
            streamer.put(best_new_tokens.cpu())

        # ---------------------------------------------------------
        # Stop conditions
        # ---------------------------------------------------------
        eos_in_best = any(int(t.item()) in eos_set for t in best_new_tokens.view(-1))
        if eos_in_best or stopping_criteria(final_input_ids, ()):
            finished = True

    # -------------------------------------------------------------------------
    # 3. Finalization
    # -------------------------------------------------------------------------
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


