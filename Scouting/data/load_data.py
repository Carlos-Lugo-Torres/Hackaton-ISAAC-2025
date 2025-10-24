import pandas as pd
from src.fingerprints.prep import (
    concat_player_season_stats,
    attach_positions,
    drop_goalkeepers,
    filter_by_minutes,
    build_fingerprint_df_autoK_FROM_DF
)

def load_clean_data(statsJugadorPorTemporada, positions_csv_path):
    # 1. Unir temporadas
    all_stats = concat_player_season_stats(statsJugadorPorTemporada, competition_id=73)

    # 2. Adjuntar posiciones
    all_stats = attach_positions(all_stats, positions_csv_path)

    # 3. Eliminar porteros
    all_stats = drop_goalkeepers(all_stats)

    # 4. Filtrar minutos mínimos
    all_stats = filter_by_minutes(all_stats, min_minutes=900)

    # 5. Generar fingerprint
    Zall, pca, scalers, info = build_fingerprint_df_autoK_FROM_DF(all_stats)

    return all_stats, Zall, pca, info