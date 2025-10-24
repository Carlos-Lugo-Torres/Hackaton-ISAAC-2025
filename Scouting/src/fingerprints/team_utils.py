import numpy as np
import pandas as pd
from numpy.linalg import norm

def similares(Zall, player_id, k=10):
    zcols = [c for c in Zall.columns if c.startswith("z_")]
    row = Zall.loc[Zall["player_id"]==player_id, zcols]
    if row.empty: raise ValueError(f"player_id {player_id} no encontrado.")
    v = row.to_numpy()[0]
    M = Zall[zcols].to_numpy()
    cos = (M @ v) / (norm(M,axis=1)*norm(v))
    out = Zall[["player_id","player_name","team_id","team_name","season_id","season_name"]].copy()
    out["cosine"] = cos
    return out[out["player_id"]!=player_id].nlargest(k, "cosine")

def get_team_members(statsJugadorPorTemporada, season_id, team_id):
    keys = [k for k in statsJugadorPorTemporada.keys() if k[0]==season_id]
    if not keys: raise ValueError(f"No hay stats para season_id={season_id}.")
    key = keys[0]
    df = statsJugadorPorTemporada[key].copy()
    rename_map = {}
    if "name" in df.columns and "player_name" not in df.columns: rename_map["name"] = "player_name"
    if "team" in df.columns and "team_name" not in df.columns: rename_map["team"] = "team_name"
    if "teamId" in df.columns and "team_id" not in df.columns: rename_map["teamId"] = "team_id"
    if "playerId" in df.columns and "player_id" not in df.columns: rename_map["playerId"] = "player_id"
    if rename_map: df = df.rename(columns=rename_map)
    if "team_id" not in df.columns: raise ValueError("No se encontró 'team_id' en player_season_stats.")
    roster = df[df["team_id"]==team_id][["player_id","player_name","team_id","team_name"]].drop_duplicates()
    return roster.sort_values("player_name").reset_index(drop=True)
