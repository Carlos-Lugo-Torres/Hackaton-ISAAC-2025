# src/fingerprints/clustering.py

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from collections import Counter

from .constants import METRICS

# ===== util =====
def _get_Xpcs(Zall, scale_pc=True, pc_cols=None):
    if pc_cols is None:
        pc_cols = [c for c in Zall.columns if c.startswith("PC")]
    X = Zall[pc_cols].to_numpy().copy()
    if scale_pc:
        X = (X - X.mean(axis=0)) / (X.std(axis=0, ddof=0) + 1e-12)
    return X

# ===== similares (tal cual tu función) =====
def similares(Zall, player_id, k=20):
    zcols = [c for c in Zall.columns if c.startswith("z_")]
    row = Zall.loc[Zall["player_id"]==player_id, zcols]
    if row.empty:
        raise ValueError(f"player_id {player_id} no encontrado.")
    v = row.to_numpy()[0]
    M = Zall[zcols].to_numpy()
    cos = (M @ v) / (norm(M,axis=1)*norm(v))
    out = Zall[["player_id","player_name","team_id","team_name","season_id","season_name"]].copy()
    out["cosine"] = cos
    return out[out["player_id"]!=player_id].nlargest(k, "cosine")

# ===== selección de k =====
def auto_select_k(Zall, k_min=2, k_max=15, *, method="silhouette",
                  scale_pc=True, pc_cols=None, random_state=42, n_init=50):
    X = _get_Xpcs(Zall, scale_pc=scale_pc, pc_cols=pc_cols)
    rows = []
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(X)
        labels = km.labels_
        rows.append({
            "k": k,
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(X, labels)) if k > 1 else float("nan"),
            "ch": float(calinski_harabasz_score(X, labels)) if k > 1 else float("nan"),
            "db": float(davies_bouldin_score(X, labels)) if k > 1 else float("nan"),
        })
    diag = pd.DataFrame(rows)

    method = method.lower()
    if method == "silhouette":
        idx = diag["silhouette"].idxmax()
        k_opt = int(diag.loc[idx, "k"])
        just = f"Máximo Silhouette = {diag.loc[idx, 'silhouette']:.3f}"
    elif method == "ch":
        idx = diag["ch"].idxmax()
        k_opt = int(diag.loc[idx, "k"])
        just = f"Máximo Calinski–Harabasz = {diag.loc[idx, 'ch']:.1f}"
    elif method == "db":
        idx = diag["db"].idxmin()
        k_opt = int(diag.loc[idx, "k"])
        just = f"Mínimo Davies–Bouldin = {diag.loc[idx, 'db']:.3f}"
    elif method == "ensemble":
        k_sil = int(diag.loc[diag["silhouette"].idxmax(), "k"])
        k_ch  = int(diag.loc[diag["ch"].idxmax(), "k"])
        k_db  = int(diag.loc[diag["db"].idxmin(), "k"])
        votos = Counter([k_sil, k_ch, k_db])
        max_votos = max(votos.values())
        candidatos = sorted([k for k, v in votos.items() if v == max_votos])
        if len(candidatos) == 1:
            k_opt = candidatos[0]
        else:
            best_sil_k = int(diag.loc[diag["silhouette"].idxmax(), "k"])
            if best_sil_k in candidatos:
                k_opt = best_sil_k
            else:
                best_ch_k = int(diag.loc[diag["ch"].idxmax(), "k"])
                if best_ch_k in candidatos:
                    k_opt = best_ch_k
                else:
                    k_opt = min(candidatos)
        just = f"Ensemble (votos: Sil={k_sil}, CH={k_ch}, DB={k_db}) → k={k_opt}"
    else:
        raise ValueError("method debe ser: 'silhouette' | 'ch' | 'db' | 'ensemble'")

    return k_opt, just, diag

def fit_kmeans_and_assign(Zall, k, *, scale_pc=True, colname=None, pc_cols=None, random_state=42, n_init=50):
    if colname is None:
        colname = f"cluster_k{k}"
    X = _get_Xpcs(Zall, scale_pc=scale_pc, pc_cols=pc_cols)
    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)
    Zall[colname] = labels
    return Zall, km

def profile_clusters(Zall, cluster_col):
    zcols = [f"z_{m}" for m in METRICS if f"z_{m}" in Zall.columns]
    grp = Zall.groupby(cluster_col, dropna=False)
    prof_z = grp[zcols].mean().sort_index()
    size = grp.size().rename("n")
    return prof_z, size

# ===== naming de roles (tal cual) =====
ROLE_RULES = [
    {"name_pos": "Mediocampista Recuperador",
     "pos": {"player_season_pressures_90","player_season_tackles_90","player_season_interceptions_90","player_season_padj_tackles_and_interceptions_90"},
     "neg": set()},
    {"name_pos": "Extremo Desequilibrante",
     "pos": {"player_season_dribbles_90","player_season_carries_90","player_season_deep_progressions_90"},
     "neg": set()},
    {"name_pos": "Creador Asociativo",
     "pos": {"player_season_key_passes_90","player_season_xa_90","player_season_lbp_completed_90","player_season_op_passes_into_box_90"},
     "neg": {"player_season_pass_length"}},
    {"name_pos": "Delantero de Área",
     "pos": {"player_season_np_shots_90","player_season_np_xg_90","player_season_touches_inside_box_90"},
     "neg": set()},
    {"name_pos": "Defensa Central Aéreo",
     "pos": {"player_season_aerial_wins_90"},
     "neg": set()},
    {"name_pos": "Lateral Progresivo",
     "pos": {"player_season_deep_progressions_90","player_season_op_passes_into_box_90"},
     "neg": set()},
]

def score_rule(cluster_mean, rule):
    s_pos = cluster_mean[[f"z_{m}" for m in rule["pos"] if f"z_{m}" in cluster_mean.index]].sum()
    s_neg = -cluster_mean[[f"z_{m}" for m in rule["neg"] if f"z_{m}" in cluster_mean.index]].sum() if rule["neg"] else 0.0
    return float(s_pos + s_neg)

def auto_name_clusters(prof_z):
    out = {}
    for cl in prof_z.index:
        row = prof_z.loc[cl]
        scored = [(score_rule(row, r), r["name_pos"]) for r in ROLE_RULES]
        scored.sort(reverse=True)
        role = scored[0][1]
        s = row.sort_values(ascending=False)
        top_pos = s.head(5).index.tolist()
        top_neg = row.sort_values(ascending=True).head(5).index.tolist()
        out[cl] = {"role": role, "top_pos": top_pos, "top_neg": top_neg}
    return out

def cluster_and_label_roles_auto(Zall, *, k_min=2, k_max=15, method="silhouette",
                                 scale_pc=True, pc_cols=None, random_state=42, n_init=50):
    k_opt, just, diag = auto_select_k(
        Zall, k_min=k_min, k_max=k_max, method=method,
        scale_pc=scale_pc, pc_cols=pc_cols, random_state=random_state, n_init=n_init
    )
    cluster_col = f"cluster_k{k_opt}"
    Zall, km = fit_kmeans_and_assign(
        Zall, k_opt, scale_pc=scale_pc, colname=cluster_col,
        pc_cols=pc_cols, random_state=random_state, n_init=n_init
    )
    prof_z, size = profile_clusters(Zall, cluster_col)
    names = auto_name_clusters(prof_z)
    Zall["Rol"] = Zall[cluster_col].map({cl: info["role"] for cl, info in names.items()})
    return Zall, km, prof_z, size, names, k_opt, just, diag
