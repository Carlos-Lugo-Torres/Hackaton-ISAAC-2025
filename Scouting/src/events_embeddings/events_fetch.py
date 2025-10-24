import pandas as pd
from statsbombpy import sb

def safe_events(match_id, creds, include_360_metrics=False):
    try:
        ev = sb.events(match_id=match_id, creds=creds, include_360_metrics=include_360_metrics)
        if ev is None or (isinstance(ev, pd.DataFrame) and ev.empty):
            print(f"[WARN] match_id={match_id}: sin eventos, lo salto.")
            return pd.DataFrame()
        return ev
    except Exception as e:
        print(f"[WARN] match_id={match_id}: error al pedir eventos ({e}). Lo salto.")
        return pd.DataFrame()

def fetch_events_for_seasons(competition_id, season_ids, creds, team_filter=None, include_360_metrics=False):
    chunks = []
    for sid in season_ids:
        matches = sb.matches(competition_id=competition_id, season_id=sid, creds=creds)
        if matches is None or matches.empty:
            print(f"[INFO] season_id={sid}: sin matches.")
            continue

        if team_filter:
            matches = matches[(matches["home_team"] == team_filter) | (matches["away_team"] == team_filter)]
            print(f"[INFO] season_id={sid}: {len(matches)} partidos del equipo '{team_filter}'.")
        else:
            print(f"[INFO] season_id={sid}: {len(matches)} partidos totales.")

        for mid in matches["match_id"].unique():
            ev = safe_events(mid, creds=creds, include_360_metrics=include_360_metrics)
            if ev.empty:
                continue
            ev["competition_id"] = competition_id
            ev["season_id"] = sid
            ev["match_id"] = mid
            chunks.append(ev)

    if not chunks:
        print("[ERROR] No se obtuvieron eventos. Revisa seasons/equipo/permisos.")
        return pd.DataFrame()

    return pd.concat(chunks, ignore_index=True)
