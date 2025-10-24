# src/fingerprints/constants.py

import numpy as np

# Métricas de la huella (tal cual tu lista)
METRICS = [
    "player_season_passing_ratio", "player_season_pass_length",
    "player_season_lbp_completed_90", "player_season_op_passes_into_box_90",
    "player_season_np_shots_90", "player_season_np_xg_90",
    "player_season_key_passes_90", "player_season_xa_90",
    "player_season_touches_inside_box_90",
    "player_season_pressures_90", "player_season_tackles_90",
    "player_season_interceptions_90", "player_season_aerial_wins_90",
    "player_season_padj_tackles_and_interceptions_90",
    "player_season_dribbles_90", "player_season_carries_90",
    "player_season_deep_progressions_90",
]

# Columnas de identificación mínimas
ID_COLS = ["player_id","player_name","team_id","team_name","competition_id","season_id","season_name"]

# Candidatas para minutos
MINUTES_CANDIDATES = ["player_season_minutes","minutes","player_minutes","player_season_360_minutes"]
