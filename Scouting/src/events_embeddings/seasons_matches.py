import pandas as pd
from statsbombpy import sb

def available_seasons(competition_id, creds):
    comps = sb.competitions(creds=creds)
    df = comps[comps["competition_id"].eq(competition_id)][
        ["competition_id", "competition_name", "season_id", "season_name"]
    ].drop_duplicates().sort_values("season_id")
    if df.empty:
        print(f"[INFO] No hay seasons para competition_id={competition_id}.")
    return df

def list_matches(competition_id, season_id, creds):
    matches = sb.matches(competition_id=competition_id, season_id=season_id, creds=creds)
    if matches is None or matches.empty:
        print(f"[INFO] Season {season_id} sin partidos o sin permisos.")
        return pd.DataFrame()
    cols = ["match_id", "match_date", "home_team", "away_team", "competition_stage"]
    cols = [c for c in cols if c in matches.columns]
    return matches[cols].sort_values("match_date")
