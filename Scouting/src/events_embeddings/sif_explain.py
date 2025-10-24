import math
import numpy as np
import pandas as pd
from collections import Counter

def _sif_setup(player_docs, a=1e-3, stop_top=35):
    tok_counts = Counter()
    total = 0
    for doc in player_docs.values:
        tok_counts.update(doc); total += len(doc)
    pw = {w: tok_counts[w]/total for w in tok_counts}

    df_counts = Counter()
    for doc in player_docs.values:
        df_counts.update(set(doc))
    stop = {w for w,_ in sorted(df_counts.items(), key=lambda x: x[1], reverse=True)[:stop_top]}
    return pw, stop

def _sif_weights(doc, pw, a=1e-3, stop=None):
    from collections import Counter as C
    c = C(doc)
    w = {}
    for tok, f in c.items():
        if stop and tok in stop:
            continue
        w[tok] = f * (a / (a + pw.get(tok, 1e-9)))
    return w

def explain_similarity(pid_a, pid_b, player_docs, a=1e-3, stop=None, pw=None, top_shared=10, top_unique=5):
    if stop is None or pw is None:
        pw, stop = _sif_setup(player_docs, a=a, stop_top=35)

    try:
        doc_a = player_docs.loc[(pid_a, next(n for (p,n) in player_docs.index if p==pid_a))]
    except StopIteration:
        doc_a = next(doc for (p,_), doc in player_docs.items() if p==pid_a)
    try:
        doc_b = player_docs.loc[(pid_b, next(n for (p,n) in player_docs.index if p==pid_b))]
    except StopIteration:
        doc_b = next(doc for (p,_), doc in player_docs.items() if p==pid_b)

    wA = _sif_weights(doc_a, pw, a=a, stop=stop)
    wB = _sif_weights(doc_b, pw, a=a, stop=stop)

    shared = []
    for tok in set(wA.keys()) & set(wB.keys()):
        shared.append((tok, wA[tok] + wB[tok], wA[tok], wB[tok]))
    shared.sort(key=lambda x: x[1], reverse=True)

    onlyA = sorted([(tok, wA[tok]) for tok in set(wA)-set(wB)], key=lambda x: x[1], reverse=True)
    onlyB = sorted([(tok, wB[tok]) for tok in set(wB)-set(wA)], key=lambda x: x[1], reverse=True)

    def base(tok): return tok.split('_',1)[0]
    def zone(tok): return '_'.join(tok.split('_')[1:3]) if tok.count('_')>=2 else tok

    out = {
        "top_shared": pd.DataFrame(
            [(tok, zone(tok), base(tok), w_sum, w_a, w_b) for tok, w_sum, w_a, w_b in shared[:top_shared]],
            columns=['token','zona','tipo','peso_total','peso_A','peso_B']
        ),
        "top_unique_A": pd.DataFrame(
            [(tok, zone(tok), base(tok), w) for tok, w in onlyA[:top_unique]],
            columns=['token','zona','tipo','peso_A']
        ),
        "top_unique_B": pd.DataFrame(
            [(tok, zone(tok), base(tok), w) for tok, w in onlyB[:top_unique]],
            columns=['token','zona','tipo','peso_B']
        )
    }
    return out

# pid2name se construye desde player_docs (tal cual lo tenías)
def show_explanation(pid_a, pid_b, player_docs, a=1e-3, stop=None, pw=None,
                     top_shared=10, top_unique=5):
    pid2name = {float(pid): pname for (pid, pname) in player_docs.index}
    expl = explain_similarity(pid_a, pid_b, player_docs, a=a, stop=stop, pw=pw,
                              top_shared=top_shared, top_unique=top_unique)
    print(f"A: {pid_a} — {pid2name.get(float(pid_a),'?')}")
    print(f"B: {pid_b} — {pid2name.get(float(pid_b),'?')}\n")
    from IPython.display import display
    print(">> Top tokens COMPARTIDOS (mayor peso combinado):")
    display(expl["top_shared"])
    print("\n>> Rasgos más distintivos de A:")
    display(expl["top_unique_A"])
    print("\n>> Rasgos más distintivos de B:")
    display(expl["top_unique_B"])
