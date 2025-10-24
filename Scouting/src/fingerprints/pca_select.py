# src/fingerprints/pca_select.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from .constants import METRICS, ID_COLS

# --- Selección automática de K (tal cual tu función) ---
def select_pca_k(Xz, method, variance_threshold=0.85,
                 cv_folds=5, random_state=42, n_iter_parallel=200):
    n_samples, n_features = Xz.shape
    max_k = min(n_samples, n_features)

    pca_full = PCA(n_components=max_k, random_state=random_state).fit(Xz)
    eigvals = pca_full.explained_variance_
    prop = pca_full.explained_variance_ratio_
    cumprop = np.cumsum(prop)

    if method == "variance":
        k = int(np.searchsorted(cumprop, variance_threshold) + 1)

    elif method == "parallel":
        rng = np.random.default_rng(random_state)
        rand_eigs = np.zeros((n_iter_parallel, len(eigvals)))
        for b in range(n_iter_parallel):
            Xb = np.empty_like(Xz)
            for j in range(n_features):
                Xb[:, j] = rng.permutation(Xz[:, j])
            pca_b = PCA(n_components=max_k, random_state=random_state).fit(Xb)
            rand_eigs[b, :len(pca_b.explained_variance_)] = pca_b.explained_variance_
        mean_rand = rand_eigs.mean(axis=0)
        k = int(np.sum(eigvals > mean_rand)) or 1

    elif method == "broken_stick":
        H = np.array([np.sum(1.0/np.arange(j, n_features+1)) for j in range(1, n_features+1)])
        bs = H / H[0]
        k = int(np.sum(prop >= bs[:len(prop)])) or 1

    elif method == "cv":
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        mse_by_k = []
        for k_try in range(1, max_k+1):
            mses = []
            pca_k = PCA(n_components=k_try, random_state=random_state)
            for tr, te in kf.split(Xz):
                pca_k.fit(Xz[tr])
                X_te_proj = pca_k.inverse_transform(pca_k.transform(Xz[te]))
                mses.append(np.mean((Xz[te] - X_te_proj)**2))
            mse_by_k.append((k_try, float(np.mean(mses))))
        k = min(mse_by_k, key=lambda t: t[1])[0]

    else:
        raise ValueError("method debe ser: 'variance' | 'parallel' | 'broken_stick' | 'cv'")

    return int(k), {
        "eigvals": eigvals,
        "prop": prop,
        "cumprop": cumprop,
        "selected_k": int(k),
        "method": method
    }

# --- Builder que parte de DF (tu FROM_DF) ---
def build_fingerprint_df_autoK_FROM_DF(df_input: pd.DataFrame,
                                       k_method="parallel",
                                       variance_threshold=0.85,
                                       group_standardization=("season_id",),
                                       random_state=42):
    df = df_input.copy()

    # Asegura columnas ID y métricas
    for c in ID_COLS:
        if c not in df.columns:
            df[c] = np.nan
    for m in METRICS:
        if m not in df.columns:
            df[m] = np.nan
        else:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    if "player_season_lbp_completed_90" in df.columns:
        df["player_season_lbp_completed_90"] = df["player_season_lbp_completed_90"].fillna(0)

    Z_parts, scaler_by_group = [], {}
    gcols = list(group_standardization) if group_standardization else []
    if gcols:
        for gkey, gdf in df.groupby(gcols, dropna=False):
            X = gdf[METRICS].copy()
            med = X.median(numeric_only=True)
            X = X.fillna(med)
            scaler = StandardScaler()
            Xz = pd.DataFrame(scaler.fit_transform(X), columns=METRICS, index=gdf.index)
            gZ = pd.concat([gdf[ID_COLS], Xz.add_prefix("z_")], axis=1)
            Z_parts.append(gZ)
            scaler_by_group[gkey if isinstance(gkey, tuple) else (gkey,)] = scaler
        Zall_new = pd.concat(Z_parts, ignore_index=True)
    else:
        X = df[METRICS].copy()
        med = X.median(numeric_only=True)
        X = X.fillna(med)
        scaler = StandardScaler()
        Xz = pd.DataFrame(scaler.fit_transform(X), columns=METRICS, index=df.index)
        Zall_new = pd.concat([df[ID_COLS], Xz.add_prefix("z_")], axis=1)
        scaler_by_group = {("GLOBAL",): scaler}

    Xz_all = Zall_new[[f"z_{m}" for m in METRICS]].to_numpy()
    k, info = select_pca_k(Xz_all, method=k_method, variance_threshold=variance_threshold, random_state=random_state)

    pca = PCA(n_components=k, random_state=random_state).fit(Xz_all)
    PCs = pca.transform(Xz_all)
    for i in range(k):
        Zall_new[f"PC{i+1}"] = PCs[:, i]

    return Zall_new, pca, scaler_by_group, info

# --- Builder que parte de dict de temporadas (tu versión original) ---
def build_fingerprint_df_autoK(statsJugadorPorTemporada, competition_id=73, min_minutes=900,
                               k_method="parallel", variance_threshold=0.85,
                               group_standardization=("season_id",), random_state=42):
    from .prep import concat_player_season_stats, filter_by_minutes

    df = concat_player_season_stats(statsJugadorPorTemporada, competition_id=competition_id)
    df = filter_by_minutes(df, min_minutes=min_minutes)
    if "player_season_lbp_completed_90" in df.columns:
        df["player_season_lbp_completed_90"] = df["player_season_lbp_completed_90"].fillna(0)

    # Reuse builder FROM_DF
    Zall, pca, scaler_by_group, info = build_fingerprint_df_autoK_FROM_DF(
        df, k_method=k_method, variance_threshold=variance_threshold,
        group_standardization=group_standardization, random_state=random_state
    )
    return Zall, pca, scaler_by_group, info
