import pandas as pd

ENGINE = "pyarrow"
SEASONS = [(317,"2024/2025"), (281,"2023/2024"), (235,"2022/2023"), (108,"2021/2022")]

statsJugadorPorTemporada = {
    (sid, sname): pd.read_parquet(f"data/Jugadores/player_season_stats_{sid}.parquet", engine=ENGINE)
    for (sid, sname) in SEASONS
}

partidos = {
    (sid, sname): pd.read_parquet(f"data/Partidos/matches_competition_73_season_{sid}.parquet", engine=ENGINE)
    for (sid, sname) in SEASONS
}

equipos = {
    (sid, sname): partidos[(sid, sname)]["home_team"].dropna().unique().tolist()
    for (sid, sname) in SEASONS
}

# src/fingerprints/io_parquet.py

from pathlib import Path
from typing import List
import pandas as pd
import numpy as np


# Columnas mínimas que intentaremos garantizar en la salida
_REQUIRED_COLS: List[str] = [
    "match_id", "timestamp", "minute", "second",
    "type",
    "player_id", "player_name",
    "team_id", "team_name",
    "location_x", "location_y",
    "under_pressure", "counterpress",
    "season_id", "competition_id"
]


def _split_location_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si el DataFrame trae una columna 'location' (lista [x, y]) y no existen
    'location_x' / 'location_y', los genera.
    """
    if "location" in df.columns and ("location_x" not in df.columns or "location_y" not in df.columns):
        def _x(v):
            try:
                return float(v[0]) if isinstance(v, (list, tuple)) and len(v) >= 2 else np.nan
            except Exception:
                return np.nan

        def _y(v):
            try:
                return float(v[1]) if isinstance(v, (list, tuple)) and len(v) >= 2 else np.nan
            except Exception:
                return np.nan

        if "location_x" not in df.columns:
            df["location_x"] = df["location"].apply(_x)
        if "location_y" not in df.columns:
            df["location_y"] = df["location"].apply(_y)
    return df


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    """Convierte a numéricas (coerce) las columnas listadas si existen."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def load_parquet_data(base_dir: Path) -> pd.DataFrame:
    """
    Lee todos los .parquet bajo base_dir (recursivo), concatena y
    devuelve un único DataFrame listo para usar en la app.

    - Si existen columnas 'location' (lista [x,y]) crea 'location_x'/'location_y'.
    - Normaliza tipos numéricos en columnas clave.
    - Asegura la presencia de columnas mínimas (crea vacías si faltan).
    """
    if not isinstance(base_dir, Path):
        base_dir = Path(base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {base_dir.resolve()}")

    # Busca todos los parquet recursivamente
    files = sorted(base_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No encontré archivos .parquet dentro de: {base_dir.resolve()}")

    chunks: List[pd.DataFrame] = []
    for p in files:
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            # Si un archivo falla, lo saltamos pero avisamos
            print(f"[WARN] No pude leer {p}: {e}")
            continue

        # Split de 'location' -> 'location_x','location_y' si aplica
        df = _split_location_columns(df)

        # Normaliza algunas columnas numéricas típicas
        _coerce_numeric(
            df,
            [
                "player_id", "team_id",
                "match_id", "minute", "second",
                "location_x", "location_y",
                "season_id", "competition_id",
            ],
        )

        chunks.append(df)

    if not chunks:
        raise RuntimeError("No se pudo cargar ningún .parquet válido.")

    events = pd.concat(chunks, ignore_index=True)

    # Asegura columnas mínimas (si faltan, las crea vacías)
    for c in _REQUIRED_COLS:
        if c not in events.columns:
            # booleans como NaN/False para flags; resto NaN
            if c in ("under_pressure", "counterpress"):
                events[c] = pd.Series([np.nan] * len(events), dtype="boolean")
            else:
                events[c] = np.nan

    # Orden de columnas amigable (las mínimas primero + el resto)
    front = [c for c in _REQUIRED_COLS if c in events.columns]
    tail = [c for c in events.columns if c not in front]
    events = events[front + tail]

    return events