import numpy as np
import pandas as pd
from .constants import METRICS, OFFENSIVE, LABEL_RULES

def top_features_for_pc(pc_name, corrs, k=6):
    s = corrs.loc[pc_name].dropna()
    top_pos = s.sort_values(ascending=False).head(k)
    top_neg = s.sort_values(ascending=True).head(k)
    return top_pos, top_neg

def orient_pc_signs(Zall, pca, corrs, offensive_set=OFFENSIVE):
    PCcols = [f"PC{i+1}" for i in range(pca.n_components_)]
    flipped = {}
    for i, pc in enumerate(PCcols):
        s = corrs.loc[pc, list(offensive_set & set(METRICS))].dropna()
        score = s.sum()
        if score < 0:
            Zall[pc] = -Zall[pc]
            corrs.loc[pc, :] = -corrs.loc[pc, :]
            pca.components_[i, :] = -pca.components_[i, :]
            flipped[pc] = True
        else:
            flipped[pc] = False
    return flipped

def score_label_rule(pc_name, corrs, rule):
    pos = corrs.loc[pc_name, list(rule["positives"] & set(METRICS))].sum() if rule["positives"] else 0.0
    neg = -corrs.loc[pc_name, list(rule["negatives"] & set(METRICS))].sum() if rule["negatives"] else 0.0
    return pos + neg

def auto_label_components(corrs, rules=LABEL_RULES):
    labels = {}
    for pc in corrs.index:
        scores = [(score_label_rule(pc, corrs, r), r["name"]) for r in rules]
        scores.sort(reverse=True)
        best = scores[0]
        labels[pc] = {"label": best[1], "score": float(best[0])}
    return labels

def pc_summary(pc, corrs, auto_labels, k=5):
    s = corrs.loc[pc].dropna()
    top_pos = s.sort_values(ascending=False).head(k)
    top_neg = s.sort_values(ascending=True).head(k)
    lab = auto_labels[pc]["label"]
    text = f"• {pc} — {lab}\n" \
           f"  Señales ↑: {', '.join([f'{m} ({top_pos[m]:+.2f})' for m in top_pos.index])}\n" \
           f"  Señales ↓: {', '.join([f'{m} ({top_neg[m]:+.2f})' for m in top_neg.index])}"
    return text

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    p,k = Phi.shape
    R = np.eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = Phi @ R
        u,s,vh = np.linalg.svd(Phi.T @ (Lambda**3 - (gamma/p) * Lambda @ np.diag(np.diag(Lambda.T @ Lambda))))
        R = u @ vh
        d = np.sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return Phi @ R, R
