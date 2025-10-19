import torch
import torch.nn as nn
import numpy as np
import os

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
            if attn is None or attn.shape != (self.n_layers, self.n_heads, -1):
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
    
    
classifier = FPAttentionClassifier(
    model_path="results_all_layers/pytorch_mlp_exp__ent1_gin0/model/pytorch_mlp_with_l2.pt",
    scaler_path="results_all_layers/pytorch_mlp_exp__ent1_gin0/model/scaler.pt",
    n_layers=32,
    n_heads=32,
    use_entropy=True,
    use_gini=False,
)

sample = {
    0: np.random.rand(32, 32, 20),
    3: np.random.rand(32, 32, 20),
    10: np.random.rand(32, 32, 20),
    15: np.random.rand(32, 32, 20),
    27: np.random.rand(32, 32, 20),
}

preds = classifier.predict(sample)
print(preds)
