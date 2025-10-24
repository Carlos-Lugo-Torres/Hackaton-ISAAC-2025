import pandas as pd
from collections import Counter

ACTION_TYPES = {'Pass','Carry','Dribble','Shot','Ball Recovery','Interception','Pressure'}

def _is_dictlike(x):
    return isinstance(x, dict)

def detect_schema(df: pd.DataFrame):
    info = {}

    if 'type' in df.columns and df['type'].notna().any() and _is_dictlike(df['type'].dropna().iloc[0]):
        info['type_mode'] = 'nested'
        info['type_name_col'] = ('type','name')
    elif 'type_name' in df.columns:
        info['type_mode'] = 'flat'
        info['type_name_col'] = 'type_name'
    else:
        info['type_mode'] = 'missing'
        info['type_name_col'] = None

    if 'player' in df.columns and df['player'].notna().any() and _is_dictlike(df['player'].dropna().iloc[0]):
        info['player_mode'] = 'nested'
        info['player_id_col']   = ('player','id')
        info['player_name_col'] = ('player','name')
    else:
        pid = 'player_id' if 'player_id' in df.columns else None
        pname = 'player_name' if 'player_name' in df.columns else ('player' if 'player' in df.columns else None)
        info['player_mode'] = 'flat' if pid or pname else 'missing'
        info['player_id_col']   = pid
        info['player_name_col'] = pname

    loc_mode = 'missing'
    if 'location' in df.columns and df['location'].notna().any():
        v = df['location'].dropna().iloc[0]
        if isinstance(v,(list,tuple)) and len(v)>=2: loc_mode = 'nested'
    if loc_mode == 'missing':
        if {'location_x','location_y'}.issubset(df.columns):
            loc_mode = 'flat_xy'
        elif {'x','y'}.issubset(df.columns):
            loc_mode = 'flat_xy_alt'
    info['location_mode'] = loc_mode

    info['has_under_pressure'] = 'under_pressure' in df.columns
    info['has_counterpress']   = 'counterpress' in df.columns

    return info

def audit_events(df: pd.DataFrame):
    print(f"Total filas: {len(df)}")
    if 'type' not in df.columns:
        raise ValueError("No existe la columna 'type' en events.")

    print("\nTop tipos de 'type':")
    print(df['type'].value_counts().head(30))

    sub = df[df['type'].isin(ACTION_TYPES)].copy()
    print(f"\nFilas con ACTION_TYPES: {len(sub)}")

    pid_col = 'player_id' if 'player_id' in df.columns else None
    pname_col = 'player_name' if 'player_name' in df.columns else ('player' if 'player' in df.columns else None)

    if pid_col:
        print(f"Con {pid_col} no nulos:", sub[pid_col].notna().sum())
    else:
        print("No encontré columna 'player_id'.")

    mask_player = sub[pid_col].notna() if pid_col else pd.Series(False, index=sub.index)
    mask_loc = sub['location'].notna() if 'location' in sub.columns else pd.Series(False, index=sub.index)
    useful = sub[mask_player & mask_loc]

    print(f"Filas con jugador y location: {len(useful)}\n")

    cols_preview = [c for c in ['match_id','timestamp','minute','second','type',
                                pid_col, pname_col, 'location','under_pressure','counterpress']
                    if c and c in df.columns]
    from IPython.display import display
    display(useful[cols_preview].head(10))
