from typing import Iterable, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm

# --- helper: obtener la fila “correcta” del jugador ---
def pick_player_row(Zall: pd.DataFrame, player_id: int, season_id: int | None = None):
    sub = Zall[Zall["player_id"] == int(player_id)].copy()
    if season_id is not None:
        sub = sub[sub["season_id"] == int(season_id)]
    if sub.empty:
        return None
    if len(sub) > 1:
        # si existe, elige la fila con más minutos
        for c in ["player_season_minutes","minutes","player_minutes","player_season_360_minutes"]:
            if c in sub.columns:
                sub = sub.sort_values(c, ascending=False)
                break
    return sub.iloc[0]

# --- 1) Definición de las facetas (puedes ajustar métricas si quieres) ---
RADAR_FACETS = {
    "Pases y Progresión": [
        "player_season_passing_ratio",
        "player_season_lbp_completed_90",
        "player_season_op_passes_into_box_90",
        "player_season_deep_progressions_90",
    ],
    "Creación/Finalización": [
        "player_season_np_shots_90",
        "player_season_np_xg_90",
        "player_season_key_passes_90",
        "player_season_xa_90",
        "player_season_touches_inside_box_90",
    ],
    "Defensiva": [
        "player_season_pressures_90",
        "player_season_tackles_90",
        "player_season_interceptions_90",
        "player_season_padj_tackles_and_interceptions_90",
    ],
    "Conducción/Dribbles": [
        "player_season_dribbles_90",
        "player_season_carries_90",
        "player_season_deep_progressions_90",
    ],
    "Dominio Aéreo": [
        "player_season_aerial_wins_90",
    ],
}

def _pick_row(Zall, player, season_id=None):
    """
    player puede ser int (player_id) o str (player_name).
    Devuelve la fila única de ese jugador (si hay varias temporadas, usa season_id o toma la de mayor minutos si existe).
    """
    df = Zall.copy()
    if season_id is not None:
        df = df[df["season_id"] == season_id]
    if isinstance(player, int):
        sub = df[df["player_id"] == player]
    else:
        # búsqueda tolerante por nombre
        ptxt = str(player).strip().lower()
        sub = df[df["player_name"].astype(str).str.lower() == ptxt]
        if sub.empty:
            sub = df[df["player_name"].astype(str).str.lower().str.contains(ptxt, na=False)]
    if sub.empty:
        raise ValueError(f"No encontré registros para '{player}' (season_id={season_id}).")
    if len(sub) > 1:
        # preferir la fila con más minutos si existe la columna
        for c in ["player_season_minutes","minutes","player_minutes","player_season_360_minutes"]:
            if c in sub.columns:
                return sub.sort_values(c, ascending=False).iloc[0]
        # si no, toma la primera
        return sub.iloc[0]
    return sub.iloc[0]

def _facet_score_from_row(row, metrics):
    """Promedio de las z-métricas disponibles en la fila; ignora las que falten."""
    zcols = [f"z_{m}" for m in metrics if f"z_{m}" in row.index]
    if not zcols:
        return np.nan
    vals = row[zcols].astype(float).values
    return np.nanmean(vals)

def _facet_scores_for_player(Zall, player, season_id=None, facets=RADAR_FACETS):
    row = _pick_row(Zall, player, season_id=season_id)
    scores = {fac: _facet_score_from_row(row, mets) for fac, mets in facets.items()}
    return scores, row  # regresa también la fila para nombre/equipo

def _minmax_scale_0_100(series_like):
    """Escala a 0–100, manejando casos constantes/NaN con 50 por defecto."""
    x = np.array(series_like, dtype=float)
    if np.all(np.isnan(x)):
        return np.full_like(x, 50.0)
    xmin = np.nanmin(x); xmax = np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or np.isclose(xmin, xmax):
        return np.full_like(x, 50.0)
    return 100 * (x - xmin) / (xmax - xmin)

def _cosine_sim(a, b):
    a = np.array(a, dtype=float); b = np.array(b, dtype=float)
    na = norm(a); nb = norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float((a @ b) / (na * nb))

def radar_compare_players(
    Zall,
    player_a, player_b,
    season_id=None,
    facets=RADAR_FACETS,
    title_prefix="Comparación de estilo (PCA → z-métricas)",
    scale_mode="dataset",  # "dataset" (recomendado) o "pair"
    fill_alpha=0.20,
    lw=2.0,
):
    """
    Crea una gráfica de radar comparando dos jugadores.
    Parámetros:
      - player_a / player_b: id (int) o nombre (str)
      - season_id: si el mismo jugador aparece en varias temporadas
      - scale_mode:
            "dataset": escala 0–100 usando distribución de TODOS los jugadores (por faceta)
            "pair":    escala 0–100 relativo SOLAMENTE a los dos jugadores (más contraste)
    """
    # 1) calcular scores por faceta
    sA, rowA = _facet_scores_for_player(Zall, player_a, season_id, facets)
    sB, rowB = _facet_scores_for_player(Zall, player_b, season_id, facets)

    # 2) construir vectores y (opcional) escalar a 0–100
    labels = list(facets.keys())
    vecA = np.array([sA[l] for l in labels], dtype=float)
    vecB = np.array([sB[l] for l in labels], dtype=float)

    if scale_mode == "dataset":
        # min/max por faceta a partir de todo el dataset
        mins, maxs = [], []
        for fac, mets in facets.items():
            zcols = [f"z_{m}" for m in mets if f"z_{m}" in Zall.columns]
            if not zcols:
                mins.append(np.nan); maxs.append(np.nan); continue
            v = Zall[zcols].mean(axis=1, numeric_only=True)
            mins.append(np.nanmin(v.values))
            maxs.append(np.nanmax(v.values))
        mins = np.array(mins, dtype=float); maxs = np.array(maxs, dtype=float)
        # proteger
        span = np.where(np.isfinite(maxs - mins) & (np.abs(maxs - mins) > 1e-9), maxs - mins, 1.0)
        radA = 100 * (vecA - mins) / span
        radB = 100 * (vecB - mins) / span
    else:
        # escala solo con A y B
        cat_min = np.nanmin(np.vstack([vecA, vecB]), axis=0)
        cat_max = np.nanmax(np.vstack([vecA, vecB]), axis=0)
        span = np.where(np.isfinite(cat_max - cat_min) & (np.abs(cat_max - cat_min) > 1e-9),
                        cat_max - cat_min, 1.0)
        radA = 100 * (vecA - cat_min) / span
        radB = 100 * (vecB - cat_min) / span

    # 3) similitud coseno (en el espacio de z-métricas base para los jugadores)
    zcols_all = [c for c in Zall.columns if c.startswith("z_")]
    baseA = _pick_row(Zall, player_a, season_id=season_id)[zcols_all].astype(float).values
    baseB = _pick_row(Zall, player_b, season_id=season_id)[zcols_all].astype(float).values
    sim = _cosine_sim(baseA, baseB)

    # 4) Radar plot
    # cierre circular
    radA = np.append(radA, radA[0])
    radB = np.append(radB, radB[0])
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles = np.append(angles, angles[0])

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    # radios fijos 0-100
    ax.set_rlabel_position(0)
    ax.set_ylim(0, 100)

    nameA = f"{rowA.get('player_name', player_a)}"
    nameB = f"{rowB.get('player_name', player_b)}"
    # traza
    ax.plot(angles, radA, linewidth=lw, label=nameA)
    ax.fill(angles, radA, alpha=fill_alpha)
    ax.plot(angles, radB, linewidth=lw, label=nameB)
    ax.fill(angles, radB, alpha=fill_alpha)

 

    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), frameon=False)
    plt.tight_layout()
    return fig

# --- CARGA DE STATS DE TEMPORADAS DESDE data/Jugadores/*.parquet ---
from pathlib import Path
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_season_stats_parquet(season_ids=None):
    """
    Busca y concatena archivos en data/Jugadores con patrón:
        player_season_stats_<season_id>.parquet
    - season_ids: iterable de ints (p.ej. [317,281]) o None para todas.
    Devuelve un DataFrame con las columnas originales + 'season_id' garantizada.
    """
    base = Path("data") / "Jugadores"
    if not base.exists():
        st.warning(f"No existe la carpeta {base.resolve()}.")
        return pd.DataFrame()

    # Detecta archivos disponibles
    found = []
    for p in sorted(base.glob("player_season_stats_*.parquet")):
        try:
            sid = int(p.stem.split("_")[-1])  # …_stats_317 -> 317
        except Exception:
            continue
        if (season_ids is None) or (sid in set(season_ids)):
            found.append((sid, p))

    if not found:
        if season_ids:
            st.error(f"No encontré parquets de stats para seasons {sorted(set(season_ids))} en {base}.")
        else:
            st.error(f"No encontré ningún parquet player_season_stats_*.parquet en {base}.")
        return pd.DataFrame()

    # Lee y concatena
    chunks = []
    for sid, path in found:
        try:
            df = pd.read_parquet(str(path), engine="pyarrow")
            if "season_id" not in df.columns:
                df["season_id"] = sid
            chunks.append(df)
        except Exception as e:
            st.error(f"Error al leer {path.name}: {e}")

    if not chunks:
        return pd.DataFrame()

    out = pd.concat(chunks, ignore_index=True)

    # Normaliza algunas columnas clave que tu pipeline usa luego
    rename_map = {}
    if "name" in out.columns and "player_name" not in out.columns: rename_map["name"] = "player_name"
    if "team" in out.columns and "team_name" not in out.columns:   rename_map["team"] = "team_name"
    if "teamId" in out.columns and "team_id" not in out.columns:   rename_map["teamId"] = "team_id"
    if "playerId" in out.columns and "player_id" not in out.columns: rename_map["playerId"] = "player_id"
    if rename_map:
        out = out.rename(columns=rename_map)

    return out



from sklearn.decomposition import PCA

def plot_embedding_scatter(
    d2v,
    highlight_pids: Tuple[float, float],
    pid2name: Optional[dict] = None,
    subset_pids: Optional[Iterable[float]] = None,
    figsize=(8, 6),
    point_size=12,
    alpha=0.45,
    random_state=42,
):
    """
    Proyecta los embeddings de Doc2Vec a 2D con PCA y dibuja un scatter,
    resaltando dos jugadores por su player_id.
    """
    # 1) Extraer todos los vectores y player_ids del Doc2Vec
    keys = []
    vecs = []
    want = set(float(pid) for pid in subset_pids) if subset_pids is not None else None

    for tag in d2v.dv.index_to_key:
        try:
            pid = float(tag)
        except Exception:
            continue
        if (want is not None) and (pid not in want):
            continue
        keys.append(pid)
        vecs.append(d2v.dv[tag])

    if not vecs:
        raise ValueError("No hay vectores para plotear (revisa subset_pids o el modelo).")

    X = np.vstack(vecs)

    # 2) PCA a 2D
    pca = PCA(n_components=2, random_state=random_state)
    XY = pca.fit_transform(X)

    coords_df = pd.DataFrame(XY, columns=["x", "y"])
    coords_df["player_id"] = keys

    # 3) Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    ax.scatter(coords_df["x"], coords_df["y"], s=point_size, alpha=alpha, linewidths=0, color="gray")

    # 4) Resaltar jugadores solicitados
    colors = ["tab:blue", "tab:orange"]
    for i, pid in enumerate(highlight_pids):
        pid_f = float(pid)
        sel = coords_df["player_id"] == pid_f
        if not sel.any():
            print(f"[WARN] player_id {pid} no está en el modelo/selección.")
            continue
        row = coords_df.loc[sel].iloc[0]
        ax.scatter([row.x], [row.y], s=120, edgecolor="black", facecolor=colors[i % 2], zorder=3)
        # Etiqueta
        base = f"{int(pid_f) if pid_f.is_integer() else pid_f}"
        label = base
        if pid2name is not None:
            name = pid2name.get(pid_f) or pid2name.get(int(pid_f), None)
            if name:
                label = f"{name} ({base})"
        ax.annotate(label, (row.x, row.y), xytext=(6, 6), textcoords="offset points",
                    fontsize=9, weight="bold")

    ax.grid(True, alpha=0.2, linestyle="--")

    evr = pca.explained_variance_ratio_
    #print(f"PCA var. explicada: PC1={evr[0]:.2%}, PC2={evr[1]:.2%}, total={evr[:2].sum():.2%}")

    return fig, ax, coords_df[["player_id", "x", "y"]]