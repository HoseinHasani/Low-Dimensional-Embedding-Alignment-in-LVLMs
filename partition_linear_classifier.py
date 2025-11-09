import os
import json
import pickle
from glob import glob
from tqdm import tqdm
import numpy as np
import joblib
from scipy.stats import ttest_ind
from collections import defaultdict
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

data_dir = "data/all layers all attention tp fp"
files_pattern = os.path.join(data_dir, "attentions_*.pkl")
files = sorted(glob(files_pattern))

class_zero = "tp"

min_position = 5
max_position = 150

win_token = 10
n_top_k = 20
n_selected_heads = 3
min_samples_per_class_window = 8

train_size = 0.75
random_state = 42

use_class_weight_balanced = True
solver = "liblinear"
max_iter = 200

base_results_dir = "results_per_window"
exp_name = f"linclf_win{win_token}_sel{n_selected_heads}_vs_{class_zero}"
results_dir = os.path.join(base_results_dir, exp_name)
os.makedirs(results_dir, exist_ok=True)

fp_replication_factor = 1

def extract_attention_values_from_file(data_dict, cls_, source="image"):
    out = []
    entries = data_dict.get(cls_, {}).get(source, [])
    for e in entries:
        subs = e.get("subtoken_results")
        if not subs:
            continue
        sub = subs[0]
        topk_vals = np.array(sub["topk_values"], dtype=float)
        if topk_vals.ndim != 3:
            continue
        idx = int(sub["idx"])
        out.append((idx, topk_vals[..., :n_top_k].astype(float)))
    return out


def collect_all_samples(files, class_zero, n_files=None):
    fps = []
    zeros = []
    n = len(files) if n_files is None else min(n_files, len(files))
    for f in tqdm(files[:n], desc="Collect files"):
        try:
            with open(f, "rb") as fh:
                data_dict = pickle.load(fh)
        except Exception as e:
            continue

        fps.extend(extract_attention_values_from_file(data_dict, "fp", source="image"))
        zeros.extend(extract_attention_values_from_file(data_dict, class_zero, source="image"))

    def to_mean_list(samples):
        out = []
        for idx, arr in samples:
            mean_arr = np.mean(arr, axis=-1)
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
        e_clipped = min(e, max_pos)
        ranges.append((s, e_clipped))
    return ranges


def gather_window_samples(samples, start, end):
    sel = [(idx, arr) for idx, arr in samples if (idx >= start and idx <= end)]
    if not sel:
        return np.zeros((0,)), []
    idxs = [s[0] for s in sel]
    arrs = np.stack([s[1] for s in sel], axis=0)
    return arrs, idxs


def perform_head_ttests(fp_arrs, zero_arrs):
    N_fp = fp_arrs.shape[0]
    N_z = zero_arrs.shape[0]
    L = fp_arrs.shape[1]
    H = fp_arrs.shape[2]
    pvals = np.ones((L, H), dtype=float)

    for l in range(L):
        for h in range(H):
            a = fp_arrs[:, l, h] if N_fp > 0 else np.array([])
            b = zero_arrs[:, l, h] if N_z > 0 else np.array([])
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
    return pvals


def build_feature_matrix(arrs, selected_coords):
    if arrs.shape[0] == 0:
        return np.zeros((0, len(selected_coords)))
    feats = np.stack([arrs[:, l, h] for (l, h) in selected_coords], axis=1)
    return feats


if __name__ == "__main__":
    print("Collecting all fp and class_zero samples (means over top_k)...")
    fps, zeros = collect_all_samples(files, class_zero)

    print(f"Total FP samples collected: {len(fps)}")
    print(f"Total {class_zero} samples collected: {len(zeros)}")

    ranges = window_ranges(min_position, max_position, win_token)
    print(f"Window ranges (start,end) to process: {ranges}")

    results = []
    overall_metrics = defaultdict(list)
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
                "start": start, "end": end, "n_fp": int(n_fp), f"n_{class_zero}": int(n_z),
                "status": "skipped_not_enough_samples"
            })
            continue

        pvals = perform_head_ttests(fp_arrs, z_arrs)
        L, H = pvals.shape
        flat_idx = np.argsort(pvals.flatten())[:n_selected_heads]
        selected_coords = [(int(idx // H), int(idx % H)) for idx in flat_idx]

        X_fp = build_feature_matrix(fp_arrs, selected_coords)
        X_z = build_feature_matrix(z_arrs, selected_coords)
        y_fp = np.ones(X_fp.shape[0], dtype=int)
        y_z = np.zeros(X_z.shape[0], dtype=int)

        X_all = np.vstack([X_fp, X_z])
        y_all = np.concatenate([y_fp, y_z])
        pos_all = np.array(fp_idxs + z_idxs)

        try:
            X_train, X_test, y_train, y_test, pos_train, pos_test = train_test_split(
                X_all, y_all, pos_all, train_size=train_size, random_state=random_state, stratify=y_all
            )
        except ValueError:
            X_train, X_test, y_train, y_test, pos_train, pos_test = train_test_split(
                X_all, y_all, pos_all, train_size=train_size, random_state=random_state
            )

        if fp_replication_factor > 1:
            fp_train_mask = (y_train == 1)
            if fp_train_mask.sum() > 0:
                X_fp_train = X_train[fp_train_mask]
                y_fp_train = y_train[fp_train_mask]
                pos_fp_train = pos_train[fp_train_mask]
                X_train = np.vstack([X_train, np.repeat(X_fp_train, fp_replication_factor - 1, axis=0)])
                y_train = np.concatenate([y_train, np.repeat(y_fp_train, fp_replication_factor - 1, axis=0)])
                pos_train = np.concatenate([pos_train, np.repeat(pos_fp_train, fp_replication_factor - 1, axis=0)])

            fp_test_mask = (y_test == 1)
            if fp_test_mask.sum() > 0:
                X_fp_test = X_test[fp_test_mask]
                y_fp_test = y_test[fp_test_mask]
                pos_fp_test = pos_test[fp_test_mask]
                X_test = np.vstack([X_test, np.repeat(X_fp_test, fp_replication_factor - 1, axis=0)])
                y_test = np.concatenate([y_test, np.repeat(y_fp_test, fp_replication_factor - 1, axis=0)])
                pos_test = np.concatenate([pos_test, np.repeat(pos_fp_test, fp_replication_factor - 1, axis=0)])

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = LogisticRegression(
            penalty="l2",
            solver=solver,
            max_iter=max_iter,
            class_weight="balanced" if use_class_weight_balanced else None,
        )

        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        y_pred_proba = clf.predict_proba(X_test_s)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "start": start,
            "end": end,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_fp": int(n_fp),
            f"n_{class_zero}": int(n_z),
            "status": "success"
        })

        overall_metrics["accuracy"].append(accuracy)
        overall_metrics["precision"].append(precision)
        overall_metrics["recall"].append(recall)
        overall_metrics["f1"].append(f1)

    with open(os.path.join(results_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    avg_metrics = {
        metric: np.mean(values) for metric, values in overall_metrics.items()
    }

    print(f"\nOverall Performance Metrics:")
    for metric, avg in avg_metrics.items():
        print(f"{metric.capitalize()}: {avg:.4f}")

    print(f"\nResults saved to: {os.path.join(results_dir, 'results.json')}")
