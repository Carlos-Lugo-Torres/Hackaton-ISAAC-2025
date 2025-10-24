import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .tokenization import _xy_from_row

_X_BINS = np.linspace(0, 120, 7)
_Y_BINS = np.linspace(0, 80, 4)
BASE_TYPES = {'Pass','Carry','Dribble','Shot','Ball Recovery','Interception','Pressure'}

def zone_matrix_6x3(events_df, player_id, base_type='Pass', normalize='row'):
    assert base_type in BASE_TYPES, f"base_type inválido: {base_type}"
    df = events_df[(events_df.get('player_id')==player_id) & (events_df['type']==base_type)].copy()
    if df.empty:
        return np.zeros((3,6)), 0

    xs, ys = [], []
    for _, r in df.iterrows():
        x,y = _xy_from_row(r)
        if x is None or y is None: 
            continue
        xs.append(x); ys.append(y)
    if not xs:
        return np.zeros((3,6)), len(df)

    H, _, _ = np.histogram2d(ys, xs, bins=[_Y_BINS, _X_BINS])
    H = H[::-1, :]

    if normalize == 'all':
        s = H.sum()
        if s > 0: H = H / s
    elif normalize == 'row':
        rs = H.sum(axis=1, keepdims=True)
        rs[rs==0] = 1.0
        H = H / rs

    return H, len(df)

def plot_zone_heatmap(H, title="", annotate=True):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    im = ax.imshow(H, aspect='auto', origin='upper')
    for c in range(7):
        ax.axvline(c-0.5, lw=0.8, alpha=0.4, color='k')
    for r in range(4):
        ax.axhline(r-0.5, lw=0.8, alpha=0.4, color='k')
    ax.set_xticks(range(6)); ax.set_yticks(range(3))
    ax.set_xticklabels(['Def-I','Def-C','Def-D','Med-I','Med-C','Med-D'], rotation=0)
    ax.set_yticklabels(['Alta','Media','Baja'])
    ax.set_title(title)
    if annotate:
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                val = H[i,j]
                txt = f"{val:.0f}" if val>=1 and val==int(val) else (f"{val*100:.0f}%" if val<=1.0 else f"{val:.0f}")
                ax.text(j, i, txt, ha='center', va='center', fontsize=9)
    plt.tight_layout()
    plt.show()

# ---- comparador por zonas ----
def _tercio(x):
    if x is None: return None
    return 'Def' if x < 40 else ('Med' if x < 80 else 'Ata')

def _carril(y):
    if y is None: return None
    return 'Izq' if y < 26.67 else ('Cen' if y < 53.33 else 'Der')

def _zone_label(x,y):
    t, c = _tercio(x), _carril(y)
    return None if (t is None or c is None) else f"{t}-{c}"

def _player_zone_counts(df, pid, base_type):
    sub = df[(df.get('player_id')==pid) & (df['type']==base_type)]
    zones = []
    for _, r in sub.iterrows():
        x,y = _xy_from_row(r)
        z = _zone_label(x,y)
        if z: zones.append(z)
    return pd.Series(zones).value_counts().rename_axis('zone').rename('count')

def compare_players_by_zone(df, pid_a, pid_b, base_type, top=None):
    assert base_type in BASE_TYPES, f"base_type inválido: {base_type}"
    ca = _player_zone_counts(df, pid_a, base_type)
    cb = _player_zone_counts(df, pid_b, base_type)

    zones = sorted(set(ca.index).union(cb.index),
                   key=lambda z: ('Def','Med','Ata').index(z.split('-')[0]) * 3 +
                                 ('Izq','Cen','Der').index(z.split('-')[1]))
    out = pd.DataFrame(index=zones)
    out['A_count'] = ca.reindex(zones).fillna(0).astype(int)
    out['B_count'] = cb.reindex(zones).fillna(0).astype(int)
    out['A_pct']   = (out['A_count'] / max(out['A_count'].sum(), 1.0)).round(4)
    out['B_pct']   = (out['B_count'] / max(out['B_count'].sum(), 1.0)).round(4)
    out['Δpct(B−A)'] = (out['B_pct'] - out['A_pct']).round(4)

    total_row = pd.DataFrame({
        'A_count':[out['A_count'].sum()],
        'B_count':[out['B_count'].sum()],
        'A_pct':[1.0 if out['A_count'].sum()>0 else 0.0],
        'B_pct':[1.0 if out['B_count'].sum()>0 else 0.0],
        'Δpct(B−A)':[out['B_pct'].sum()-out['A_pct'].sum()]
    }, index=['TOTAL'])

    if top:
        core = out.reindex(out.index, copy=True)
        core = core.iloc[np.argsort(-core['Δpct(B−A)'].abs().values)]
        core = core.head(top)
        out = pd.concat([core, total_row])
    else:
        out = pd.concat([out, total_row])

    disp = out.copy() 
    for c in ['A_pct','B_pct','Δpct(B−A)']:
        disp[c] = (disp[c]*100).round(1).astype(str) + '%'
    disp = disp.reset_index().rename(columns={'index':'zone'})
    return disp
