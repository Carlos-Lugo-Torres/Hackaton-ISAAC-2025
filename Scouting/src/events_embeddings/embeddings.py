# src/events_embeddings/embeddings.py
# ============================================================
# Embeddings (versión compatible con la app)
# - Usa location_x/location_y (tu versión nueva)
# - Mantiene wrappers con nombres antiguos: build_player_docs, event_to_token_plus, zona_token, tercio, carril
# - No entrena ni ejecuta nada al importarse
# ============================================================

import os
from pathlib import Path
import math
from collections import Counter

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.decomposition import PCA

from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# ------------------------------------------------------------
# Configuración y constantes
# ------------------------------------------------------------
pd.set_option("display.max_columns", 200)

# Tipos de acción que definen estilo
ACTION_TYPES = {'Pass', 'Carry', 'Dribble', 'Shot', 'Ball Recovery', 'Interception', 'Pressure'}

# Ruta base de ejemplo para lectura por carpetas (opcional; útil en scripts offline)
BASE_EVENTS = Path("Datos") / "Eventos"   # cambia si tu ruta es distinta


# ------------------------------------------------------------
# BLOQUE 1: Lectura de eventos desde carpetas season_*
# ------------------------------------------------------------
def listar_temp_y_archivos(base_dir: Path = BASE_EVENTS):
    """
    Devuelve lista [(season_id, [paths parquet...]), ...] ordenada por season.
    Espera estructura: base_dir/season_XXXX/events_*.parquet
    """
    items = []
    if not base_dir.exists():
        raise FileNotFoundError(f"No existe {base_dir.resolve()}")
    for d in sorted(base_dir.glob("season_*")):
        if not d.is_dir():
            continue
        try:
            season_id = int(str(d.name).split("_")[-1])
        except Exception:
            continue
        files = sorted(d.glob("events_*.parquet"))
        if files:
            items.append((season_id, files))
    return items


def cargar_eventos(base_dir: Path = BASE_EVENTS, seasons: list[int] | None = None) -> pd.DataFrame:
    """
    Lee todos los .parquet encontrados.
    - seasons: lista de season_id a incluir (None = todas)
    Retorna: DataFrame concatenado.
    """
    season_files = listar_temp_y_archivos(base_dir)
    if seasons is not None:
        season_files = [(sid, fls) for sid, fls in season_files if sid in set(seasons)]
    if not season_files:
        raise FileNotFoundError(f"No encontré archivos en {base_dir}/season_/events_.parquet")

    chunks = []
    for sid, files in season_files:
        for p in files:
            df = pd.read_parquet(p)
            # aseguremos algunas columnas clave
            for col in ['player_id', 'type', 'match_id', 'minute', 'second', 'location_x', 'location_y']:
                if col not in df.columns:
                    df[col] = pd.NA
            df['season_id'] = sid  # por si faltara
            chunks.append(df)

    events = pd.concat(chunks, ignore_index=True)

    # tipos/orden mínimo
    numeric_cols = ['player_id', 'match_id', 'minute', 'second', 'location_x', 'location_y', 'season_id', 'competition_id']
    for c in numeric_cols:
        if c in events.columns:
            events[c] = pd.to_numeric(events[c], errors='coerce')

    # Auditoría rápida (stdout)
    print(f"Total filas: {len(events):,}")
    if 'type' in events.columns:
        print("\nTop tipos de 'type':")
        print(events['type'].value_counts().head(20))
    keep = ['match_id', 'timestamp', 'minute', 'second', 'type', 'player_id', 'location_x', 'location_y', 'under_pressure', 'counterpress']
    keep = [c for c in keep if c in events.columns]
    print("\n=== PREVIEW COLUMNAS DISPONIBLES ===")
    print("\t".join(keep))
    print(events[keep].head(10))

    # métrica útil para tokenización
    sub = events[events['type'].isin(ACTION_TYPES)].copy()
    n_action = len(sub)
    n_pid = sub['player_id'].notna().sum()
    print(f"\nFilas con ACTION_TYPES: {n_action:,}")
    print(f"Con player_id no nulos: {n_pid:,}")

    return events


# ------------------------------------------------------------
# BLOQUE 2: Tokenización y documentos por jugador (XY)
# ------------------------------------------------------------
def _tercio(x: float | None) -> str | None:
    if x is None:
        return None
    return 'Def' if x < 40 else ('Med' if x < 80 else 'Ata')


def _carril(y: float | None) -> str | None:
    if y is None:
        return None
    return 'Izq' if y < 26.67 else ('Cen' if y < 53.33 else 'Der')


def zona_token_from_xy(row: dict | pd.Series) -> str:
    x = row.get('location_x')
    y = row.get('location_y')
    if pd.isna(x) or pd.isna(y):
        return 'Zona_NA'
    return f"Z_{tercio(float(x))}{_carril(float(y))}"


def _is_true(x) -> bool:
    return isinstance(x, (bool, np.bool_)) and bool(x)


def _get(row: dict | pd.Series, col: str):
    return row[col] if (col in row and pd.notna(row[col])) else None


def event_to_token_plus_xy(row: dict | pd.Series) -> str | None:
    """
    Construye la 'palabra' combinando:
      - type (Pass/Carry/Dribble/Shot/Ball Recovery/Interception/Pressure)
      - zona 3x3 a partir de location_x/location_y
      - modificadores disponibles (under_pressure, counterpress, pass/shot/dribble attrs)
    """
    tname = row['type']
    if tname not in ACTION_TYPES:
        return None

    ztok = zona_token_from_xy(row)
    mods = []
    if _is_true(row.get('under_pressure')): mods.append('BajoPresion')
    if _is_true(row.get('counterpress')):   mods.append('ContraPresion')

    if tname == 'Pass':
        h  = _get(row, 'pass_height_name')
        bp = _get(row, 'pass_body_part_name')
        oc = _get(row, 'pass_outcome_name')
        if h:  mods.append({'Ground Pass':'Raso', 'Low Pass':'Bajo', 'High Pass':'Alto'}.get(h, h.replace(' ', '_')))
        if bp: mods.append({'Right Foot':'PieDer', 'Left Foot':'PieIzq', 'Head':'Cabeza'}.get(bp, bp.replace(' ', '_')))
        if oc: mods.append('Out_' + oc.replace(' ', ''))
    elif tname == 'Shot':
        tech = _get(row, 'shot_technique_name')
        bp   = _get(row, 'shot_body_part_name')
        ft   = _get(row, 'shot_first_time')
        if _is_true(ft): mods.append('PrimerToque')
        if tech: mods.append(tech.replace(' ', ''))
        if bp:   mods.append({'Right Foot':'PieDer', 'Left Foot':'PieIzq', 'Head':'Cabeza'}.get(bp, bp.replace(' ', '_')))
    elif tname == 'Dribble':
        oc = _get(row, 'dribble_outcome_name')
        if oc: mods.append(oc)

    return f"{tname}{ztok}" + (f"{'_'.join(mods)}" if mods else "")


def build_player_docs_basic_xy(df: pd.DataFrame, min_actions: int = 30, tok_fn=event_to_token_plus_xy) -> pd.Series:
    """
    Crea documentos (lista de tokens) por player_id.
    - Ordena por (match_id, minute, second) si existen
    - Agrupa SOLO por player_id (player_name puede venir NaN). Asigna nombre seguro.
    - Filtra jugadores con al menos min_actions tokens.
    Retorna: Serie con índice MultiIndex (player_id, player_name) -> list(tokens)
    """
    by = [c for c in ('match_id', 'minute', 'second') if c in df.columns]
    df = df.sort_values(by)

    rows = []
    it = df.itertuples(index=False, name=None)
    cols = list(df.columns)
    idx_pid = cols.index('player_id') if 'player_id' in cols else None
    idx_type = cols.index('type') if 'type' in cols else None
    idx_pname = cols.index('player_name') if 'player_name' in cols else None

    for row in it:
        r = dict(zip(cols, row))
        if idx_pid is None or idx_type is None:
            continue
        if pd.isna(r['player_id']):
            continue
        tok = tok_fn(r)
        if tok:
            pname = r['player_name'] if idx_pname is not None else None
            rows.append((r['player_id'], pname, tok))

    if not rows:
        print("[INFO] No se generaron tokens (revisa 'type', 'player_id' y location_x/y).")
        return pd.Series(dtype=object)

    tok_df = pd.DataFrame(rows, columns=['player_id', 'player_name', 'token'])

    # Agrupa por player_id; toma el primer nombre no nulo como etiqueta
    agg = tok_df.groupby('player_id').agg(
        tokens=('token', list),
        name=('player_name', lambda s: next((x for x in s if pd.notna(x)), None))
    )

    # nombre seguro si faltó
    agg['player_name'] = [
        (f"player_{int(pid)}" if (pd.isna(nm) or nm is None) else str(nm))
        for pid, nm in zip(agg.index, agg['name'])
    ]
    agg.drop(columns=['name'], inplace=True)

    # filtra por mínimo de acciones
    agg = agg[agg['tokens'].apply(len) >= min_actions]

    # Serie con índice MultiIndex (player_id, player_name)
    docs = pd.Series(
        agg['tokens'].values,
        index=pd.MultiIndex.from_arrays([agg.index, agg['player_name']], names=['player_id', 'player_name'])
    )
    return docs


def quick_diag(docs: pd.Series, topn: int = 20):
    """Diagnóstico rápido del corpus de documentos por jugador."""
    vocab = Counter()
    for d in docs.values:
        vocab.update(d)
    print(f"# jugadores: {len(docs)}")
    lens = [len(d) for d in docs.values]
    if lens:
        print(f"tokens/jugador -> min:{min(lens)} p50:{np.median(lens):.0f} max:{max(lens)} mean:{np.mean(lens):.1f}")
    print(f"# vocab: {len(vocab)} | Top {topn}:")
    for tok, c in vocab.most_common(topn):
        print(f"{tok:55s} {c}")


# ------------------------------------------------------------
# BLOQUE 3: Entrenamiento de embeddings (para uso offline)
# ------------------------------------------------------------
def train_embeddings(player_docs: pd.Series,
                     w2v_dim: int = 128,
                     d2v_dim: int = 128,
                     window: int = 8,
                     min_count: int = 1,
                     workers: int = 4,
                     epochs: int = 12):
    """
    Entrena Word2Vec y Doc2Vec a partir de player_docs (serie de listas de tokens).
    Devuelve (w2v, d2v).
    """
    corpus = list(player_docs.values)
    if not corpus:
        raise ValueError("Corpus vacío: no hay documentos de jugadores.")

    total_tokens = int(np.sum([len(d) for d in corpus]))
    print(f"Entrenando Word2Vec con {len(corpus)} documentos y {total_tokens:,} tokens...")

    w2v = Word2Vec(
        sentences=corpus,
        vector_size=w2v_dim,
        window=window,
        min_count=min_count,
        sg=1,             # Skip-gram
        negative=10,
        workers=workers,
        epochs=epochs
    )

    tagged = [TaggedDocument(words=doc, tags=[str(pid)]) for (pid, _), doc in player_docs.items()]
    print(f"Entrenando Doc2Vec ({len(tagged)} jugadores)...")

    d2v = Doc2Vec(
        documents=tagged,
        vector_size=d2v_dim,
        window=window,
        min_count=min_count,
        dm=1,            # PV-DM
        negative=10,
        workers=workers,
        epochs=epochs
    )

    print("✅ Entrenamiento completado.")
    return w2v, d2v


# ------------------------------------------------------------
# BLOQUE 4: Vecinos Doc2Vec (nativo)
# ------------------------------------------------------------
def doc2vec_neighbors(target_pid: float, d2v: Doc2Vec, k: int = 10) -> pd.DataFrame:
    """
    Top-K más similares usando el índice nativo de Doc2Vec.
    Si existe un mapa global pid2name, lo usa; si no, deja nombre como None.
    """
    key = str(target_pid)
    try:
        sims = d2v.dv.most_similar(key, topn=k + 20)  # pido más por si sale el propio
    except KeyError:
        raise ValueError(f"Doc2Vec no tiene la clave {key}. Revisa que el player_id exista en player_docs.")
    rows = []
    # soporte opcional a pid2name global si el usuario lo define fuera
    try:
        name_map = globals().get("pid2name", {})
    except Exception:
        name_map = {}
    for tag, sim in sims:
        if tag == key:  # omite el propio
            continue
        try:
            pid = float(tag)
        except Exception:
            pid = tag
        rows.append((pid, name_map.get(pid, None), float(sim)))
        if len(rows) >= k:
            break
    return pd.DataFrame(rows, columns=['player_id', 'player_name', 'similaridad'])


# ------------------------------------------------------------
# WRAPPERS DE COMPATIBILIDAD (para la app)
# ------------------------------------------------------------
def tercio(x):
    """Alias del tercio original (para compatibilidad)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    return _tercio(float(x))


def carril(y):
    """Alias del carril original (para compatibilidad)."""
    if y is None or (isinstance(y, float) and pd.isna(y)):
        return None
    return _carril(float(y))


def zona_token(x, y):
    """
    Firma antigua: zona a partir de (x,y).
    Internamente usa la misma lógica que tu versión XY.
    """
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return "Zona_NA"
    return f"Z_{tercio(x)}_{carril(y)}"


def event_to_token_plus(row):
    """
    Wrapper con el nombre original esperado por la app.
    Reutiliza tu versión nueva basada en location_x/location_y.
    """
    return event_to_token_plus_xy(row)


def build_player_docs(df: pd.DataFrame, min_actions: int = 30, tok_fn=event_to_token_plus) -> pd.Series:
    """
    Wrapper con la firma original.
    Internamente usa tu implementación nueva basada en XY.
    """
    _tok = tok_fn if tok_fn is not None else event_to_token_plus
    return build_player_docs_basic_xy(df, min_actions=min_actions, tok_fn=_tok)

