"""

Saves:
 - model and scaler per window directory:
     results/{exp_name}/window_{start}_{end}/
       - model.joblib
       - scaler.joblib
       - metadata.json
 - overall results CSV: results/{exp_name}/window_results_summary.csv

Assumptions about your pickle files (as described by you):
 - Each file is a dict with keys 'tp', 'fp', 'other' (or other classes)
 - data_dict[cls]['image'] is a list of samples
 - each sample has 'subtoken_results' where each sub has:
     - 'idx' : integer token position in generated text
     - 'topk_values' : shape (n_layers, n_heads, n_top_k)
"""

import os
import json
import pickle
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import joblib
from scipy.stats import ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import pandas as pd

# ---------------------------
# User-tweakable parameters
# ---------------------------
data_dir = "data/all layers all attention tp fp"  # directory with attentions_*.pkl
files_pattern = os.path.join(data_dir, "attentions_*.pkl")
files = sorted(glob(files_pattern))

# which class is zero (the negative class). 'fp' is always positive (1).
class_zero = "tp"   # or 'other' etc.

# positions (only consider tokens inside [min_position, max_position])
min_position = 5
max_position = 150

# windowing over token positions
win_token = 5  # tokens [start, start + win_token] inclusive; step = win_token
# e.g., start=0 -> window [0,5], next start=5 -> [5,10] (overlap at boundary token)

# attention reduction
n_top_k = 20  # average over first K visual tokens

# model selection
n_selected_heads = 7   # number of (layer,head) features to select per window
min_samples_per_class_window = 8  # skip windows with fewer samples per class than this

# training / splitting
train_size = 0.75
random_state = 42

# linear classifier settings
use_class_weight_balanced = True
solver = "liblinear"  # good for small datasets
max_iter = 200

# results saving
base_results_dir = "results_per_window"
exp_name = f"linclf_win{win_token}_sel{n_selected_heads}_vs_{class_zero}"
results_dir = os.path.join(base_results_dir, exp_name)
os.makedirs(results_dir, exist_ok=True)

# optional: if you'd like to replicate fp samples instead of class_weight, set factor>1.
# replication will be applied only to the training fold (not test).
fp_replication_factor = 1  # set >1 if you want explicit replication (slower). Default 1 -> no replication.
# ---------------------------

# helper functions
def extract_attention_values_from_file(data_dict, cls_, source="image"):
    """Return a list of (idx, topk_arr) for the given class and source.
       topk_arr shape: (n_layers, n_heads, n_top_k)
    """
    out = []
    entries = data_dict.get(cls_, {}).get(source, [])
    for e in entries:
        subs = e.get("subtoken_results")
        if not subs:
            continue
        # handle only first subtoken (consistent with your earlier code n_subtokens=1)
        sub = subs[0]
        topk_vals = np.array(sub["topk_values"], dtype=float)
        if topk_vals.ndim != 3:
            continue
        idx = int(sub["idx"])
        out.append((idx, topk_vals[..., :n_top_k].astype(float)))
    return out


def collect_all_samples(files, class_zero, n_files=None):
    """
    Collect samples for 'fp' and class_zero across files.
    Returns two lists:
      fps : list of (idx, mean_att_array)  mean_att_array shape (n_layers, n_heads)
      zeros: same for class_zero
    """
    fps = []
    zeros = []
    n = len(files) if n_files is None else min(n_files, len(files))
    for f in tqdm(files[:n], desc="Collect files"):
        try:
            with open(f, "rb") as fh:
                data_dict = pickle.load(fh)
        except Exception as e:
            # skip unreadable files
            # print(f"skip {f}: {e}")
            continue

        fps.extend(extract_attention_values_from_file(data_dict, "fp", source="image"))
        zeros.extend(extract_attention_values_from_file(data_dict, class_zero, source="image"))

    # Convert each topk array -> mean over top_k to get (n_layers, n_heads)
    def to_mean_list(samples):
        out = []
        for idx, arr in samples:
            # arr shape (n_layers, n_heads, n_top_k)
            mean_arr = np.mean(arr, axis=-1)  # shape (n_layers, n_heads)
            out.append((int(idx), mean_arr.astype(float)))
        return out

    fps = to_mean_list(fps)
    zeros = to_mean_list(zeros)
    return fps, zeros


def window_ranges(min_pos, max_pos, win_token):
    starts = list(range(min_pos, max_pos + 1, win_token))
    ranges = []
    for s in starts:
        e = s + win_token
        if e < min_pos:
            continue
        if s > max_pos:
            continue
        # clip end to max_pos
        e_clipped = min(e, max_pos)
        ranges.append((s, e_clipped))
    return ranges


def gather_window_samples(samples, start, end):
    """Given list of (idx, mean_arr), return arrays of shape (n_samples, n_layers, n_heads) and list of idxs."""
    sel = [ (idx,arr) for idx,arr in samples if (idx >= start and idx <= end) ]
    if not sel:
        return np.zeros((0,)), []
    idxs = [s[0] for s in sel]
    arrs = np.stack([s[1] for s in sel], axis=0)  # shape (n_samples, n_layers, n_heads)
    return arrs, idxs


def perform_head_ttests(fp_arrs, zero_arrs):
    """
    fp_arrs : (N_fp, L, H)
    zero_arrs : (N_z, L, H)
    returns pvals_flat : flattened array shape (L*H,) with p-values (higher significance -> smaller p)
    and indices mapping flattened idx -> (l,h)
    """
    N_fp = fp_arrs.shape[0]
    N_z = zero_arrs.shape[0]
    L = fp_arrs.shape[1]
    H = fp_arrs.shape[2]
    pvals = np.ones((L, H), dtype=float)

    # For each head do Welch t-test. If insufficient samples, p=1.
    for l in range(L):
        for h in range(H):
            a = fp_arrs[:, l, h] if N_fp > 0 else np.array([])
            b = zero_arrs[:, l, h] if N_z > 0 else np.array([])
            # require at least 2 samples in each to t-test
            if len(a) < 2 or len(b) < 2:
                p = 1.0
            else:
                try:
                    t, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
                    if np.isnan(p):
                        p = 1.0
                except Exception:
                    p = 1.0
            pvals[l, h] = p
    return pvals  # (L,H)


# ---------------------------
# Main pipeline
# ---------------------------
if __name__ == "__main__":
    print("Collecting all fp and class_zero samples (means over top_k)...")
    fps, zeros = collect_all_samples(files, class_zero)

    print(f"Total FP samples collected: {len(fps)}")
    print(f"Total {class_zero} samples collected: {len(zeros)}")

    # Build windows
    ranges = window_ranges(min_position, max_position, win_token)
    print(f"Window ranges (start,end) to process: {ranges}")

    results = []
    for (start, end) in ranges:
        print(f"\nProcessing window [{start}, {end}] ...")
        fp_arrs, fp_idxs = gather_window_samples(fps, start, end)
        z_arrs, z_idxs = gather_window_samples(zeros, start, end)

        n_fp = fp_arrs.shape[0]
        n_z = z_arrs.shape[0]
        print(f"  Samples in window: FP={n_fp}, {class_zero}={n_z}")

        if n_fp < min_samples_per_class_window or n_z < min_samples_per_class_window:
            print(f"  Skipping window [{start},{end}] (not enough samples).")
            results.append({
                "start":start, "end":end, "n_fp":int(n_fp), f"n_{class_zero}":int(n_z),
                "status":"skipped_not_enough_samples"
            })
            continue

        # compute p-values per head
        pvals = perform_head_ttests(fp_arrs, z_arrs)  # shape (L,H)
        L, H = pvals.shape
        flat_idx = np.argsort(pvals.flatten())[:n_selected_heads]  # indices of selected heads
        selected_coords = [ (int(idx // H), int(idx % H)) for idx in flat_idx ]
        selected_pvals = [ float(pvals[l,h]) for (l,h) in selected_coords ]
        print(f"  Selected (layer,head) coords (top {n_selected_heads} by p-value): {selected_coords}")
        print(f"  Selected p-values: {selected_pvals}")

        # build dataset for this window
        # each sample -> feature vector of selected heads in specified order
        def build_feature_matrix(arrs):
            # arrs shape (N, L, H) -> selected features shape (N, n_selected_heads)
            if arrs.shape[0] == 0:
                return np.zeros((0, len(selected_coords)))
            feats = np.stack([ arrs[:, l, h] for (l,h) in selected_coords ], axis=1)
            return feats

        X_fp = build_feature_matrix(fp_arrs)  # (n_fp, k)
        X_z = build_feature_matrix(z_arrs)    # (n_z, k)
        y_fp = np.ones(X_fp.shape[0], dtype=int)
        y_z = np.zeros(X_z.shape[0], dtype=int)

        X_all = np.vstack([X_fp, X_z])
        y_all = np.concatenate([y_fp, y_z])
        pos_all = np.array(fp_idxs + z_idxs)

        # split into train/test - stratified
        try:
            X_train, X_test, y_train, y_test, pos_train, pos_test = train_test_split(
                X_all, y_all, pos_all, train_size=train_size, random_state=random_state, stratify=y_all
            )
        except ValueError:
            # fallback: non-stratified if stratify fails (rare)
            X_train, X_test, y_train, y_test, pos_train, pos_test = train_test_split(
                X_all, y_all, pos_all, train_size=train_size, random_state=random_state
            )

        # optional replication of FP in training fold (if desired)
        if fp_replication_factor > 1:
            fp_train_mask = (y_train == 1)
            if fp_train_mask.sum() > 0:
                X_fp_train = X_train[fp_train_mask]
                y_fp_train = y_train[fp_train_mask]
                pos_fp_train = pos_train[fp_train_mask]
                X_train = np.vstack([X_train, np.repeat(X_fp_train, fp_replication_factor-1, axis=0)])
                y_train = np.concatenate([y_train, np.repeat(y_fp_train, fp_replication_factor-1, axis=0)])
                pos_train = np.concatenate([pos_train, np.repeat(pos_fp_train, fp_replication_factor-1, axis=0)])

        # Normalize features (fit on train only)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train linear classifier (Logistic Regression)
        clf = LogisticRegression(
            penalty="l2",
            solver=solver,
            max_iter=max_iter,
            class_weight="balanced" if use_class_weight_balanced else None,
            random_state=random_state,
        )
        clf.fit(X_train_s, y_train)

        # Evaluate
        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        print(f"  Eval (test): acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")

        # Save model & scaler & metadata
        win_dir = os.path.join(results_dir, f"window_{start}_{end}")
        os.makedirs(win_dir, exist_ok=True)
        model_path = os.path.join(win_dir, f"linclf_{class_zero}_vs_fp_sel{n_selected_heads}.joblib")
        scaler_path = os.path.join(win_dir, f"scaler_{class_zero}_vs_fp_sel{n_selected_heads}.joblib")
        meta_path = os.path.join(win_dir, "metadata.json")

        joblib.dump(clf, model_path)
        joblib.dump(scaler, scaler_path)

        metadata = {
            "class_zero": class_zero,
            "window_start": int(start),
            "window_end": int(end),
            "n_selected_heads": int(n_selected_heads),
            "selected_coords": selected_coords,
            "selected_pvals": selected_pvals,
            "n_fp_total_window": int(n_fp),
            f"n_{class_zero}_total_window": int(n_z),
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "scaler_path": scaler_path,
            "model_path": model_path,
            "use_class_weight_balanced": bool(use_class_weight_balanced),
            "fp_replication_factor": int(fp_replication_factor),
        }
        with open(meta_path, "w") as fh:
            json.dump(metadata, fh, indent=2)

        # Append results summary
        results.append({
            "start": start,
            "end": end,
            "n_fp": int(n_fp),
            f"n_{class_zero}": int(n_z),
            "train_n": int(X_train.shape[0]),
            "test_n": int(X_test.shape[0]),
            "acc": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "selected_coords": selected_coords,
            "selected_pvals": selected_pvals,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "status": "trained",
        })

    # Save summary CSV and JSON
    summary_df = pd.DataFrame(results)
    csv_path = os.path.join(results_dir, "window_results_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    json_path = os.path.join(results_dir, "window_results_summary.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\nAll done.")
    print(f"Results saved to: {results_dir}")
    print(f"Summary CSV: {csv_path}")
