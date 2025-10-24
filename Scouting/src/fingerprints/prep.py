# src/fingerprints/prep.py

import numpy as np
import pandas as pd
from .constants import METRICS, ID_COLS, MINUTES_CANDIDATES

# ==== utilidades ====
def _first_existing(colnames, candidates):
    for c in candidates:
        if c in colnames:
            return c
    return None

# Normalización etiqueta de posición
GK_SYNONYMS = {"goalkeeper", "gk", "portero", "arquero", "guardameta", "goal keeper"}

def _norm_pos(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace("-", " ").replace("_", " ")
    return s

# ==== preparación de datos ====
def concat_player_season_stats(statsJugadorPorTemporada, competition_id=73):
    frames = []
    for (season_id, season_name), df in statsJugadorPorTemporada.items():
        tmp = df.copy()

        if "competition_id" not in tmp.columns: tmp["competition_id"] = competition_id
        if "season_id" not in tmp.columns: tmp["season_id"] = season_id
        if "season_name" not in tmp.columns: tmp["season_name"] = season_name

        rename_map = {}
        if "name" in tmp.columns and "player_name" not in tmp.columns: rename_map["name"] = "player_name"
        if "team" in tmp.columns and "team_name" not in tmp.columns:   rename_map["team"] = "team_name"
        if "teamId" in tmp.columns and "team_id" not in tmp.columns:   rename_map["teamId"] = "team_id"
        if "playerId" in tmp.columns and "player_id" not in tmp.columns: rename_map["playerId"] = "player_id"
        if rename_map:
            tmp = tmp.rename(columns=rename_map)

        frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    all_stats = pd.concat(frames, axis=0, ignore_index=True)

    # Asegura columnas base
    for c in ["team_id","team_name"]:
        if c not in all_stats.columns:
            all_stats[c] = np.nan

    # Asegura métricas (coerción numérica; si faltan, NaN)
    for m in METRICS:
        if m in all_stats.columns:
            all_stats[m] = pd.to_numeric(all_stats[m], errors="coerce")
        else:
            all_stats[m] = np.nan

    # LBP a 0 si falta
    if "player_season_lbp_completed_90" not in all_stats.columns:
        all_stats["player_season_lbp_completed_90"] = 0.0

    return all_stats

def filter_by_minutes(df, min_minutes=900):
    min_col = _first_existing(df.columns, MINUTES_CANDIDATES)
    if min_col is None:
        return df.copy()
    return df[df[min_col] >= min_minutes].copy()

def attach_positions(df_stats: pd.DataFrame, pos_csv_path: str) -> pd.DataFrame:
    """
    Une posiciones (season_id, player_id, position) y crea position_norm.
    """
    pos = pd.read_csv(pos_csv_path, usecols=["season_id", "player_id", "position"])
    pos["position_norm"] = pos["position"].map(_norm_pos)
    out = df_stats.merge(
        pos[["season_id", "player_id", "position_norm"]],
        on=["season_id", "player_id"],
        how="left"
    )
    return out

def drop_goalkeepers(df_stats_with_pos: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con posición conocida de portero."""
    return df_stats_with_pos.loc[~df_stats_with_pos["position_norm"].isin(GK_SYNONYMS)].copy()

def get_team_members(statsJugadorPorTemporada, season_id, team_id):
    keys = [k for k in statsJugadorPorTemporada.keys() if k[0]==season_id]
    if not keys:
        raise ValueError(f"No hay stats para season_id={season_id}.")
    key = keys[0]
    df = statsJugadorPorTemporada[key].copy()

    rename_map = {}
    if "name" in df.columns and "player_name" not in df.columns: rename_map["name"] = "player_name"
    if "team" in df.columns and "team_name" not in df.columns:   rename_map["team"] = "team_name"
    if "teamId" in df.columns and "team_id" not in df.columns:   rename_map["teamId"] = "team_id"
    if "playerId" in df.columns and "player_id" not in df.columns: rename_map["playerId"] = "player_id"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "team_id" not in df.columns:
        raise ValueError("No se encontró 'team_id' en player_season_stats.")

    roster = df[df["team_id"]==team_id][["player_id","player_name","team_id","team_name"]].drop_duplicates()
    return roster.sort_values("player_name").reset_index(drop=True)
