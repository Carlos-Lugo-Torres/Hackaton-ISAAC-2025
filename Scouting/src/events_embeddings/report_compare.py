import numpy as np
import pandas as pd
from numpy.linalg import norm
from .zones import BASE_TYPES, compare_players_by_zone

def _cos(a,b):
    return float(np.dot(a, b) / (norm(a)*norm(b) + 1e-12))

def generate_player_comparison_report(pid_a, pid_b, events, player_docs, sif_pool=None, d2v=None, top_zones=9):
    pid2name = {float(pid): pname for (pid, pname) in player_docs.index}
    name_a = pid2name.get(float(pid_a), str(pid_a))
    name_b = pid2name.get(float(pid_b), str(pid_b))

    sim_sif = None
    if sif_pool and pid_a in sif_pool and pid_b in sif_pool:
        sim_sif = _cos(sif_pool[pid_a][1], sif_pool[pid_b][1])

    sim_d2v = None
    if d2v is not None and str(pid_a) in d2v.dv and str(pid_b) in d2v.dv:
        sim_d2v = _cos(d2v.dv[str(pid_a)], d2v.dv[str(pid_b)])

    meta = pd.DataFrame({
        'metric':['SIF_cosine','Doc2Vec_cosine'],
        'value':[sim_sif, sim_d2v]
    })

    zone_tables = {}
    for bt in BASE_TYPES:
        zone_tables[bt] = compare_players_by_zone(events, pid_a, pid_b, base_type=bt, top=top_zones)

    explanation = None
    try:
        from .sif_explain import explain_similarity
        explanation = explain_similarity(pid_a, pid_b, player_docs, top_shared=10, top_unique=5)
    except Exception:
        explanation = None

    report = {
        'players': {'A': {'id': pid_a, 'name': name_a}, 'B': {'id': pid_b, 'name': name_b}},
        'similarities': meta,
        'zones': zone_tables,
        'explanation': explanation
    }

    from IPython.display import display
    print(f"Comparativo: A={name_a} ({pid_a})  vs  B={name_b} ({pid_b})")
    display(meta)
    for bt in BASE_TYPES:
        print(f"\n== {bt} ==")
        display(zone_tables[bt])
    if explanation:
        print("\n>> Top tokens compartidos")
        display(explanation['top_shared'])
        print("\n>> Rasgos distintivos de A")
        display(explanation['top_unique_A'])
        print("\n>> Rasgos distintivos de B")
        display(explanation['top_unique_B'])

    return report
