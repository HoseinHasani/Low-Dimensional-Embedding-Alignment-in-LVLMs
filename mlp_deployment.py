import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from glob import glob
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


class MLPClassifierTorch(nn.Module):
    def __init__(self, input_dim, hidden1=512, hidden2=64, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class FPAttentionClassifier:
    def __init__(self, model_path, scaler_path,
                 n_layers=32, n_heads=32, max_position=150,
                 use_entropy=True, use_gini=True, dropout_rate=0.5):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_position = max_position
        self.use_entropy = use_entropy
        self.use_gini = use_gini

        scaler_data = torch.load(scaler_path, map_location=self.device)
        self.mean = scaler_data["mean"].to(self.device)
        self.std = scaler_data["std"].to(self.device)

        input_dim = self.mean.numel()

        self.model = MLPClassifierTorch(input_dim, dropout_rate=dropout_rate).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def compute_entropy(self, values):
        eps = 1e-10
        vals = np.array(values, dtype=float)
        probs = vals / (np.sum(vals, axis=-1, keepdims=True) + eps)
        return -np.sum(probs * np.log(probs + eps), axis=-1)

    def compute_gini(self, values):
        eps = 1e-10
        vals = np.array(values, dtype=float)
        probs = vals / (np.sum(vals, axis=-1, keepdims=True) + eps)
        sorted_p = np.sort(probs, axis=-1)
        n = sorted_p.shape[-1]
        coef = 2 * np.arange(1, n + 1) - n - 1
        gini = np.sum(coef * sorted_p, axis=-1) / (n * np.sum(sorted_p, axis=-1) + eps)
        return np.abs(gini)

    def _extract_features_for_token(self, token_idx, attn):
        """attn: numpy array (n_layers, n_heads, top_k)"""
        features = []
        for l in range(self.n_layers):
            for h in range(self.n_heads):
                vals = attn[l, h, :]
                mean_val = np.mean(vals)
                features.append(mean_val)
                if self.use_entropy:
                    features.append(self.compute_entropy(vals))
                if self.use_gini:
                    features.append(self.compute_gini(vals))
                    
        features.append(token_idx / self.max_position)
        return np.array(features, dtype=np.float32)

    def predict(self, attention_dict):
        all_features = []
        token_indices = []

        for token_idx, attn in attention_dict.items():
            if attn is None or attn.shape[0] != self.n_layers or attn.shape[1] != self.n_heads:
                continue
            feats = self._extract_features_for_token(token_idx, attn)
            all_features.append(feats)
            token_indices.append(token_idx)

        if not all_features:
            return {}

        X = torch.tensor(np.stack(all_features), dtype=torch.float32, device=self.device)
        X = (X - self.mean) / (self.std + 1e-6)
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.sigmoid(logits).cpu().numpy()
            
        

        return {idx: float(p) for idx, p in zip(token_indices, probs)}
    

    


def load_real_attention_samples(data_dir, n_samples=100, n_layers=32, n_heads=32, n_top_k=20):
    files = sorted(glob(os.path.join(data_dir, "attentions_*.pkl")))
    samples = []

    for fpath in files[:n_samples]:
        try:
            with open(fpath, "rb") as handle:
                data = pickle.load(handle)
        except Exception as e:
            print(f"Skipping {fpath}: {e}")
            continue

        for cls_name, label in [("fp", 1), ("tp", 0), ("other", 0)]:
            entries = data.get(cls_name, {}).get("image", [])
            for e in entries:
                if not e.get("subtoken_results"):
                    continue
                for sub in e["subtoken_results"][:1]:
                    topk_vals = np.array(sub.get("topk_values"), dtype=float)
                    if topk_vals.ndim != 3:
                        continue
                    idx = int(sub.get("idx", -1))
                    if idx < 0:
                        continue

                    topk_vals = topk_vals[..., :n_top_k]
                    attn_dict = {idx: topk_vals}
                    samples.append((attn_dict, label))
    return samples



def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()




def evaluate_on_real_data(classifier, data_dir, n_eval=100):
    samples = load_real_attention_samples(data_dir, n_samples=n_eval,
                                          n_layers=classifier.n_layers,
                                          n_heads=classifier.n_heads)

    y_true, y_pred = [], []

    for attn_dict, label in tqdm(samples):
        preds = classifier.predict(attn_dict)
        if not preds:
            continue
        prob = list(preds.values())[0]
        y_true.append(label)
        y_pred.append(prob)

    if not y_true:
        print("No valid samples found for evaluation.")
        return

    print(np.max(y_pred), np.min(y_pred), np.mean(y_pred), np.std(y_pred))

    y_bin = (np.array(y_pred) > 0.5).astype(int)
    acc = accuracy_score(y_true, y_bin)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_bin, average="binary"
    )

    print(f"\n=== Evaluation on {n_eval} Data ===")
    print(f"Samples:   {len(y_true)}")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-score:  {f1:.3f}")

    plot_confusion_matrix(y_true, y_bin, classes=['Not Target', 'Target'], normalize=False)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


    
if __name__ == "__main__":
    
    classifier = FPAttentionClassifier(
        model_path="results_all_layers/pytorch_mlp_exp__ent1_gin0/model/pytorch_mlp_with_l2.pt",
        scaler_path="results_all_layers/pytorch_mlp_exp__ent1_gin0/model/scaler.pt",
        n_layers=32,
        n_heads=32,
        use_entropy=True,
        use_gini=False,
    )

    # Dummy attention data for testing
    sample = {
        0: np.random.rand(32, 32, 20),
        3: np.random.rand(32, 32, 20),
        10: np.random.rand(32, 32, 20),
        20: np.random.rand(32, 32, 20),
    }

    preds = classifier.predict(sample)
    print("\nPredictions for a dummy sample:")
    for idx, prob in preds.items():
        print(f"Token {idx:>3}: {prob:.4f}")
        
    if False:
        data_dir = "data/all layers all attention tp fp"
        print("\nEvaluating classifier on real dataset samples...")
        results = evaluate_on_real_data(classifier, data_dir, n_eval=3000)
