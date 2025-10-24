import math
import numpy as np
import pandas as pd
from collections import Counter
from numpy.linalg import norm

ACTION_TYPES = {'Pass','Carry','Dribble','Shot','Ball Recovery','Interception','Pressure'}

def tercio(x):  return 'Def' if x < 40 else ('Med' if x < 80 else 'Ata')
def carril(y):  return 'Izq' if y < 26.67 else ('Cen' if y < 53.33 else 'Der')
def zona_token(x, y):
    if x is None or y is None: return 'Zona_NA'
    return f"Z_{tercio(float(x))}_{carril(float(y))}"

def _is_true(x):
    return isinstance(x, (bool, np.bool_)) and bool(x)

def _get(row, col):
    return row[col] if (col in row and pd.notna(row[col])) else None

def cosine_sim(a,b):
    return float(np.dot(a,b)/(norm(a)*norm(b)+1e-12))

def _xy_from_row(row):
    loc = row.get('location', None)
    if isinstance(loc, (list,tuple)) and len(loc)>=2:
        return float(loc[0]), float(loc[1])
    return None, None

def event_to_token_plus(row):
    tname = row['type']
    if tname not in ACTION_TYPES: 
        return None
    x, y = _xy_from_row(row)
    ztok = zona_token(x, y)
    mods = []
    if _is_true(row.get('under_pressure')): mods.append('BajoPresion')
    if _is_true(row.get('counterpress')):   mods.append('ContraPresion')

    if tname == 'Pass':
        h  = _get(row, 'pass_height_name')
        bp = _get(row, 'pass_body_part_name')
        oc = _get(row, 'pass_outcome_name')
        if h:  mods.append({'Ground Pass':'Raso','Low Pass':'Bajo','High Pass':'Alto'}.get(h, h.replace(' ','_')))
        if bp: mods.append({'Right Foot':'PieDer','Left Foot':'PieIzq','Head':'Cabeza'}.get(bp, bp.replace(' ','_')))
        if oc: mods.append('Out_'+oc.replace(' ',''))
    elif tname == 'Shot':
        tech = _get(row, 'shot_technique_name')
        bp   = _get(row, 'shot_body_part_name')
        ft   = _get(row, 'shot_first_time')
        if _is_true(ft): mods.append('PrimerToque')
        if tech: mods.append(tech.replace(' ',''))
        if bp:   mods.append({'Right Foot':'PieDer','Left Foot':'PieIzq','Head':'Cabeza'}.get(bp, bp.replace(' ','_')))
    elif tname == 'Dribble':
        oc = _get(row, 'dribble_outcome_name')
        if oc: mods.append(oc)

    return f"{tname}_{ztok}" + (f"_{'_'.join(mods)}" if mods else "")

def build_player_docs(df, min_actions=30, tok_fn=event_to_token_plus):
    by = [c for c in ('match_id','minute','second') if c in df.columns]
    df = df.sort_values(by)
    rows = []
    for _, r in df.iterrows():
        tok = tok_fn(r)
        if tok:
            pid = r.get('player_id')
            pname = r.get('player_name', r.get('player'))
            if pd.notna(pid):
                rows.append((pid, pname, tok))
    if not rows:
        return pd.Series(dtype=object)
    tok_df = pd.DataFrame(rows, columns=['player_id','player_name','token'])
    docs = tok_df.groupby(['player_id','player_name'])['token'].apply(list)
    return docs[docs.apply(len) >= min_actions]

def quick_diag(docs, topn=20):
    vocab = Counter()
    for d in docs.values: vocab.update(d)
    print(f"# jugadores: {len(docs)} | # vocab: {len(vocab)}")
    for tok, c in vocab.most_common(topn):
        print(f"{tok:50s} {c}")

def find_players(name_substr, players_index, top=10):
    m = players_index[players_index['player_name'].str.contains(name_substr, case=False, na=False)]
    return m.head(top)
