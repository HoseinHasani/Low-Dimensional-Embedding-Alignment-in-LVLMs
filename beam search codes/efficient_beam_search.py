import nltk
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
import re

import re
from typing import List

def find_llava_indices(gen_text: str, chair_tokens: List[List[str]], subwords) -> List[List[int]]:

    results = []

    # --- Build offsets by walking the text sequentially ---
    offsets = []
    pointer = 0
    text_lower = gen_text.lower()
    for sw in subwords:
        sw_clean = sw.strip()
        if not sw_clean:
            offsets.append((pointer, pointer))
            continue

        # Find sw starting from current pointer
        idx = text_lower.find(sw_clean.lower(), pointer)
        if idx == -1:
            start, end = pointer, pointer + len(sw_clean)
        else:
            start, end = idx, idx + len(sw_clean)
        offsets.append((start, end))
        pointer = end

    # --- Collect matches for each word ---
    matches_by_word = {}
    for pair in chair_tokens:
        target = pair[0]
        key = target.lower()
        if key not in matches_by_word:
            pattern = rf"\b{re.escape(target)}(es|s)?\b"
            matches_by_word[key] = [m.span() for m in re.finditer(pattern, gen_text, flags=re.IGNORECASE)]

    consumed_idx = {k: 0 for k in matches_by_word}

    # --- Sequentially assign matches ---
    for pair in chair_tokens:
        target = pair[0]
        key = target.lower()
        spans = matches_by_word.get(key, [])
        idx = consumed_idx[key]

        if idx >= len(spans):
            results.append([])
            continue

        start_char, end_char = spans[idx]

        # Collect all token indices overlapping this match
        indices = [
            i for i, (tok_start, tok_end) in enumerate(offsets)
            if tok_start < end_char and tok_end > start_char
        ]

        results.append(indices)
        consumed_idx[key] += 1

    return results



def extract_top_attentions_by_steps(all_attentions, token_indices, image_token_index, image_tokens_count,
                                    topk=5, layer_start=0, layer_end=14, aggregate=True):

    results = {}
    batch_size = all_attentions[0][0].shape[0]
    assert batch_size == 1, "Only batch size 1 supported."

    n_layers_total = len(all_attentions[0])
    layer_end = min(layer_end, n_layers_total - 1)
    
    token_indices = [inds[0] for inds in token_indices if len(inds) > 0]

    for idx in token_indices:

        # (num_layers, B, H, S, S)
        try:
            attn_matrix = all_attentions[idx][layer_start:layer_end + 1]
        except:
            print("Error",idx)
            
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
    tokenizer,
    output_ids=None,
    image_token_index=35,
    image_tokens_count=576,
    topk=20,
    layer_start=0,
    layer_end=32,
    use_nltk=False,  
):
    res_len = len(candidate_attentions)
    if use_nltk:
        generated_text = tokenizer.batch_decode(
            output_ids[:, -len(candidate_attentions):], skip_special_tokens=True
        )[0]
        subwords = [tokenizer.decode([tid]) for tid in output_ids[0][-(res_len+10):]]
        subwords = subwords[-res_len:]

        words = nltk.word_tokenize(generated_text.lower())
        tagged_sent = nltk.pos_tag(words)
        nouns = [[word,word] for word, tag in tagged_sent if "NN" in tag]

        noun_indices = find_llava_indices(generated_text, nouns, subwords)
        all_indices = noun_indices

    else:
        all_indices = [[i] for i in range(res_len)]

    all_attn = extract_top_attentions_by_steps(
        candidate_attentions,
        all_indices,
        image_token_index,
        image_tokens_count,
        topk=topk,
        layer_start=layer_start,
        layer_end=layer_end,
    )
    n_fp = [0]
    if len(all_attn)>0:
        shifted_attn = {k + token_offset: v for k, v in all_attn.items()}
        preds = classifier.predict(shifted_attn)


        for k, p in preds.items():
            if 3 <= k <= 160:
                if p > 0.5:
                    n_fp.extend([1])

    return -np.sum(n_fp)




def _clone_past_key_values(past_key_values):
    """Clone a tuple of past_key_values tensors safely (detach + clone)."""
    if past_key_values is None:
        return None
    return tuple(tuple(p.detach().clone() for p in layer) for layer in past_key_values)

def _detach_and_clone_model_kwargs(model_kwargs, device=None):
    """Shallow copy model_kwargs but deep-clone its tensor values."""
    mk = {}
    for k, v in model_kwargs.items():
        if k == "past_key_values":
            mk[k] = _clone_past_key_values(v)
        elif isinstance(v, torch.Tensor):
            mk[k] = v.detach().clone()
        elif isinstance(v, (list, tuple)):
            mk[k] = type(v)(
                (item.detach().clone() if isinstance(item, torch.Tensor) else item)
                for item in v
            )
        else:
            mk[k] = copy.deepcopy(v)
    if device is not None and "past_key_values" in mk and mk["past_key_values"] is not None:
        mk["past_key_values"] = tuple(
            tuple(t.to(device) for t in layer) for layer in mk["past_key_values"]
        )
    return mk

def _assert_cache_alignment(model_kwargs, input_ids):
    """Check that cached sequence length matches current token count."""
    pk = model_kwargs.get("past_key_values", None)
    if pk is None:
        return
    cached_len = pk[0][0].shape[-2]
    if cached_len != input_ids.shape[1]:
        print(f"[WARN] Cache length {cached_len} != input tokens {input_ids.shape[1]}")



# --- add this helper near sentence_evaluator (or replace sentence_evaluator usage) ---
def count_hallucinations(
    candidate_attentions,
    classifier,
    token_offset,
    tokenizer,
    output_ids=None,
    image_token_index=35,
    image_tokens_count=576,
    topk=20,
    layer_start=0,
    layer_end=32,
    use_nltk=False,
    threshold: float = 0.5,
):
    """
    Return the number of hallucinated tokens (integer) for a (partial) candidate.
    Mirrors the logic of sentence_evaluator but returns a positive count and
    works on partial sequences.
    """
    res_len = len(candidate_attentions)
    if res_len == 0:
        return 0

    # When using nltk flow: compute noun indices; otherwise, treat every generated token as candidate
    if use_nltk:
        # We need output_ids (the partial generated token IDs) to do token->word alignment.
        if output_ids is None:
            # If no output ids available, conservative fallback: treat as zero hallucinations.
            return 0
        generated_text = tokenizer.batch_decode(
            output_ids[:, -res_len:], skip_special_tokens=True
        )[0]
        subwords = [tokenizer.decode([tid]) for tid in output_ids[0][-(res_len+10):]]
        subwords = subwords[-res_len:]

        words = nltk.word_tokenize(generated_text.lower())
        tagged_sent = nltk.pos_tag(words)
        nouns = [[word, word] for word, tag in tagged_sent if "NN" in tag]
        all_indices = find_llava_indices(generated_text, nouns, subwords)
    else:
        all_indices = [[i] for i in range(res_len)]

    all_attn = extract_top_attentions_by_steps(
        candidate_attentions,
        all_indices,
        image_token_index,
        image_tokens_count,
        topk=topk,
        layer_start=layer_start,
        layer_end=layer_end,
    )

    if len(all_attn) == 0:
        return 0

    shifted_attn = {k + token_offset: v for k, v in all_attn.items()}
    preds = classifier.predict(shifted_attn)

    n_fp = 0
    for k, p in preds.items():
        # keep the same positional filter as before
        if 3 <= k <= 160:
            if p > threshold:
                n_fp += 1
    return n_fp


# --- Replace or augment sample_with_ensemble with this updated implementation ---
def sample_with_ensemble(
    self,
    input_ids: torch.LongTensor,
    ensemble_size: int = 10,
    pseudo_sentence_length: int = 20,
    search_start: int = 2,
    diversity_win_length: int = 10,
    diversity_random_thresh: float = 0.3,
    apply_diversity: bool = False,
    use_nltk: bool = False,
    adaptive: bool = True,  # NEW: whether to use efficient adaptive beam search
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
):
    """
    Ensemble-guided sampling with optional efficient adaptive beam search.
    If adaptive=True we generate the first candidate fully and then generate
    subsequent candidates token-by-token with live hallucination scoring and early pruning.
    """
    classifier = getattr(self, "classifier", None)
    tokenizer = getattr(self, "tokenizer", None)
    # ---------- Setup ----------
    logits_processor = logits_processor or LogitsProcessorList()
    logits_warper = logits_warper or LogitsProcessorList()
    stopping_criteria = stopping_criteria or StoppingCriteriaList()
    if max_length is not None:
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

    pad_token_id = pad_token_id or self.generation_config.pad_token_id
    eos_token_id = eos_token_id or self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device)

    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # ---------- Initialize ----------
    final_input_ids = input_ids.clone()
    model_kwargs_main = model_kwargs.copy()
    finished = False
    total_generated = 0

    # ---------- Phase 1: Warm-up ----------
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
        if next_tokens[0].item() in eos_token_id:
            finished = True
        if streamer is not None:
            streamer.put(next_tokens.cpu())

    # ---------- Main ensemble/adaptive loop ----------
    while not finished:
        # storage for the ensemble round (we will pick one candidate to append)
        best_candidate = None
        best_candidate_model_kwargs = None
        best_candidate_attentions = None
        best_hall_count = None

        # --- Candidate 0: fully generate and use as baseline ---
        model_kwargs_tmp = _detach_and_clone_model_kwargs(model_kwargs_main, device=next(self.parameters()).device)
        input_tmp = final_input_ids.clone()
        decoder_attn_tmp = []
        token_count, eos_hit = 0, False
        diverse_enough = True

        while token_count < pseudo_sentence_length and not eos_hit:
            model_inputs = self.prepare_inputs_for_generation(input_tmp, **model_kwargs_tmp)
            outputs = self(**model_inputs, return_dict=True, output_attentions=True)

            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = logits_processor(input_tmp, next_token_logits)
            next_token_scores = logits_warper(input_tmp, next_token_scores)
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            input_tmp = torch.cat([input_tmp, next_tokens[:, None]], dim=-1)
            decoder_attn_tmp.append(
                outputs.cross_attentions if self.config.is_encoder_decoder else outputs.attentions
            )
            model_kwargs_tmp = self._update_model_kwargs_for_generation(
                outputs, model_kwargs_tmp, is_encoder_decoder=self.config.is_encoder_decoder
            )

            token_count += 1
            if next_tokens[0].item() in eos_token_id:
                eos_hit = True

            if apply_diversity and token_count == diversity_win_length:
                prefix = tuple(input_tmp[0, -diversity_win_length:].tolist())
                # we'll decide diversity acceptance after loop
                # (we still record prefix for later)
                # do not break here—keep behavior consistent with original code
                # duplicate check will be applied below

        # Diversity check for candidate 0
        if apply_diversity:
            prefix0 = tuple(input_tmp[0, -diversity_win_length:].tolist())
        else:
            prefix0 = None

        # Record baseline
        best_candidate = input_tmp
        best_candidate_attentions = decoder_attn_tmp
        best_candidate_model_kwargs = model_kwargs_tmp
        # compute hallucination count for baseline candidate
        n_h_star = count_hallucinations(
            best_candidate_attentions,
            classifier,
            total_generated,
            tokenizer,
            output_ids=best_candidate,
            use_nltk=use_nltk,
        )
        best_hall_count = n_h_star

        # If adaptive disabled, fallback to original exhaustive method: collect other candidates fully and score them
        if not adaptive:
            candidate_outputs, candidate_attentions, candidate_model_kwargs = [best_candidate], [best_candidate_attentions], [best_candidate_model_kwargs]
            prefix_list = [prefix0] if prefix0 is not None else []

            # generate remaining ensemble_size-1 candidates fully (original behavior)
            for candidate_idx in range(1, ensemble_size):
                model_kwargs_tmp = _detach_and_clone_model_kwargs(model_kwargs_main, device=next(self.parameters()).device)
                input_tmp = final_input_ids.clone()
                decoder_attn_tmp = []
                token_count, eos_hit = 0, False
                diverse_enough = True

                while token_count < pseudo_sentence_length and not eos_hit:
                    model_inputs = self.prepare_inputs_for_generation(input_tmp, **model_kwargs_tmp)
                    outputs = self(**model_inputs, return_dict=True, output_attentions=True)

                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_scores = logits_processor(input_tmp, next_token_logits)
                    next_token_scores = logits_warper(input_tmp, next_token_scores)
                    probs = F.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

                    input_tmp = torch.cat([input_tmp, next_tokens[:, None]], dim=-1)
                    decoder_attn_tmp.append(
                        outputs.cross_attentions if self.config.is_encoder_decoder else outputs.attentions
                    )
                    model_kwargs_tmp = self._update_model_kwargs_for_generation(
                        outputs, model_kwargs_tmp, is_encoder_decoder=self.config.is_encoder_decoder
                    )

                    token_count += 1
                    if next_tokens[0].item() in eos_token_id:
                        eos_hit = True

                if apply_diversity:
                    prefix = tuple(input_tmp[0, -diversity_win_length:].tolist())
                    duplicate_found = any(prefix == prev for prev in prefix_list)
                    if duplicate_found and torch.rand(1).item() >= diversity_random_thresh:
                        # skip candidate
                        continue
                    prefix_list.append(prefix)

                candidate_outputs.append(input_tmp)
                candidate_attentions.append(decoder_attn_tmp)
                candidate_model_kwargs.append(model_kwargs_tmp)

            # Evaluate all fully generated candidates (use sentence_evaluator to be consistent)
            candidate_scores = []
            for i in range(len(candidate_outputs)):
                candidate_scores.append(
                    sentence_evaluator(candidate_attentions[i], classifier, total_generated, tokenizer, candidate_outputs[i], use_nltk=use_nltk)
                )
            total_generated += pseudo_sentence_length
            best_idx = int(torch.tensor(candidate_scores).argmax())

            final_input_ids = candidate_outputs[best_idx]
            model_kwargs_main = _detach_and_clone_model_kwargs(candidate_model_kwargs[best_idx], device=next(self.parameters()).device)
            _assert_cache_alignment(model_kwargs_main, final_input_ids)

            best_tokens = candidate_outputs[best_idx][:, -pseudo_sentence_length:]
            if any(t.item() in eos_token_id for t in best_tokens[0]) or stopping_criteria(final_input_ids, None):
                finished = True
            if streamer is not None:
                streamer.put(best_tokens.cpu())
            continue  # next outer loop iteration

        # ---------- Adaptive mode: use baseline and prune subsequent candidates live ----------
        # if the baseline is perfect (no hallucinations), accept it immediately
        if best_hall_count == 0:
            chosen_candidate = best_candidate
            chosen_candidate_model_kwargs = best_candidate_model_kwargs
            chosen_candidate_attentions = best_candidate_attentions
        else:
            # Try remaining candidates one by one; prune them early based on partial hallucination count
            prefix_list = [prefix0] if prefix0 is not None else []
            chosen_candidate = best_candidate
            chosen_candidate_model_kwargs = best_candidate_model_kwargs
            chosen_candidate_attentions = best_candidate_attentions

            for candidate_idx in range(1, ensemble_size):
                model_kwargs_tmp = _detach_and_clone_model_kwargs(model_kwargs_main, device=next(self.parameters()).device)
                input_tmp = final_input_ids.clone()
                decoder_attn_tmp = []
                token_count, eos_hit = 0, False
                diverse_enough = True
                aborted = False

                while token_count < pseudo_sentence_length and not eos_hit:
                    model_inputs = self.prepare_inputs_for_generation(input_tmp, **model_kwargs_tmp)
                    outputs = self(**model_inputs, return_dict=True, output_attentions=True)

                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_scores = logits_processor(input_tmp, next_token_logits)
                    next_token_scores = logits_warper(input_tmp, next_token_scores)
                    probs = F.softmax(next_token_scores, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

                    input_tmp = torch.cat([input_tmp, next_tokens[:, None]], dim=-1)
                    decoder_attn_tmp.append(
                        outputs.cross_attentions if self.config.is_encoder_decoder else outputs.attentions
                    )
                    model_kwargs_tmp = self._update_model_kwargs_for_generation(
                        outputs, model_kwargs_tmp, is_encoder_decoder=self.config.is_encoder_decoder
                    )

                    token_count += 1
                    if next_tokens[0].item() in eos_token_id:
                        eos_hit = True

                    # Live scoring: compute hallucination count on the partial candidate so far
                    partial_h = count_hallucinations(
                        decoder_attn_tmp,
                        classifier,
                        total_generated,
                        tokenizer,
                        output_ids=input_tmp,
                        use_nltk=use_nltk,
                    )

                    # prune early if this partial hallucination count already exceeds baseline
                    if partial_h > best_hall_count:
                        aborted = True
                        break

                # diversity check (if enabled)
                if apply_diversity:
                    prefix = tuple(input_tmp[0, -diversity_win_length:].tolist())
                    duplicate_found = any(prefix == prev for prev in prefix_list)
                    if duplicate_found and torch.rand(1).item() >= diversity_random_thresh:
                        continue
                    prefix_list.append(prefix)

                if aborted:
                    # prune; move to next candidate
                    continue

                # Candidate finished normally: compute final hallucination count
                final_h = count_hallucinations(
                    decoder_attn_tmp,
                    classifier,
                    total_generated,
                    tokenizer,
                    output_ids=input_tmp,
                    use_nltk=use_nltk,
                )

                if final_h < best_hall_count:
                    # improvement: adopt as new best and tighten bound
                    chosen_candidate = input_tmp
                    chosen_candidate_model_kwargs = model_kwargs_tmp
                    chosen_candidate_attentions = decoder_attn_tmp
                    best_hall_count = final_h
                    # If we reach 0, we cannot do better; terminate early.
                    if best_hall_count == 0:
                        break

            # end loop over candidates

        # After ensemble/adaptive pass: accept chosen_candidate
        final_input_ids = chosen_candidate
        model_kwargs_main = _detach_and_clone_model_kwargs(chosen_candidate_model_kwargs, device=next(self.parameters()).device)
        _assert_cache_alignment(model_kwargs_main, final_input_ids)

        total_generated += pseudo_sentence_length

        best_tokens = final_input_ids[:, -pseudo_sentence_length:]
        if any(t.item() in eos_token_id for t in best_tokens[0]) or stopping_criteria(final_input_ids, None):
            finished = True

        if streamer is not None:
            streamer.put(best_tokens.cpu())

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        return SampleDecoderOnlyOutput(sequences=final_input_ids, attentions=None, hidden_states=None)
    else:
        return final_input_ids


# --- hook into transformers as before ---
def evolve_beam_search():
    transformers.generation.utils.GenerationMixin.sample = sample_with_ensemble
    transformers.generation.utils.GenerationMixin._sample = sample_with_ensemble

