

import pickle
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import streamlit as st

# ======================================================
# Utilidades de ruta robustas (ancladas al repo)
# ======================================================
def _project_root() -> Path:
    # .../src/modules/frankenstein.py -> repo root
    return Path(__file__).resolve().parents[2]

def _default_paths():
    root = _project_root()
    # Ajusta estas rutas si tus carpetas tienen otro nombre
    docs = f"data/Modelos/player_docs_20251023-145024.pkl.gz"
    w2v  = f"data/Modelos/word2vec_128d_20251023-145024.model"   # opcional
    return docs, w2v

# ======================================================
# Carga de datos/modelos (cacheadas)
# ======================================================
@st.cache_data(show_spinner=True)
def load_player_docs(path: str) -> pd.Series:
    p = Path(path)
    with open(p, "rb") as f:
        player_docs = pickle.load(f)
    # Esperamos un objeto parecido a: index=(player_id, player_name) -> list[str]
    return player_docs

@st.cache_resource(show_spinner=True)
def train_w2v_from_docs(player_docs, dim=128, window=8, min_count=1, epochs=12, workers=4):
    corpus = list(player_docs.values)
    w2v = Word2Vec(
        sentences=corpus, vector_size=dim, window=window,
        min_count=min_count, sg=1, negative=10, workers=workers, epochs=epochs
    )
    return w2v

def build_w2v_pool(player_docs, w2v) -> dict[str, tuple[str, np.ndarray]]:
    vecs = {}
    for (pid, pname), doc in player_docs.items():
        emb = [w2v.wv[w] for w in doc if w in w2v.wv]
        if not emb:
            continue
        vecs[str(pid)] = (pname, np.mean(emb, axis=0))
    return vecs

def build_w2v_sif(player_docs, w2v, a=1e-3, stop_top=35, remove_pc=True):
    tok_counts = Counter()
    total_tokens = 0
    for doc in player_docs.values:
        tok_counts.update(doc)
        total_tokens += len(doc)
    pw = {w: tok_counts[w] / total_tokens for w in tok_counts}

    # tokens más frecuentes por DF
    df_counts = Counter()
    for doc in player_docs.values:
        df_counts.update(set(doc))
    stop = {w for w, _ in sorted(df_counts.items(), key=lambda x: x[1], reverse=True)[:stop_top]}

    pid_list, name_list, vec_list = [], [], []
    for (pid, pname), doc in player_docs.items():
        wvecs, weights = [], []
        for w in doc:
            if w in stop or w not in w2v.wv:
                continue
            weight = a / (a + pw.get(w, 1e-9))
            wvecs.append(w2v.wv[w]); weights.append(weight)
        if wvecs:
            v = np.average(wvecs, axis=0, weights=weights)
            pid_list.append(str(pid)); name_list.append(pname); vec_list.append(v)

    if not vec_list:
        return {}

    V = np.vstack(vec_list)
    if remove_pc and V.shape[0] >= 2:
        pca = PCA(n_components=1).fit(V)
        u = pca.components_[0]
        V = V - (V @ u[:, None]) * u[None, :]
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return {pid: (name_list[i], V[i]) for i, pid in enumerate(pid_list)}

def synthesize_embedding(components, *, space, sif_pool=None, w2v_pool=None, normalize=True):
    pools = {"sif": sif_pool, "w2v": w2v_pool}
    pool = pools.get(space)
    if pool is None:
        raise ValueError(f"space debe ser 'sif' o 'w2v'; recibido: {space}")
    vec = np.zeros_like(next(iter(pool.values()))[1])
    usados = []
    for pid, weight in components:
        key = str(pid)
        if key in pool:
            usados.append(key)
            vec += weight * pool[key][1]
    if normalize:
        vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec, usados

def plot_projection_2d_highlight(pool, vec_syn=None, chosen_pids=None,
                                 sample=600, annotate=True, random_state=42):
    chosen_pids = [] if chosen_pids is None else [str(p) for p in chosen_pids]

    items = list(pool.items())
    if len(items) > sample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(items), size=sample, replace=False)
        items = [items[i] for i in idx]

    keep = {pid for pid, _ in items}
    for pid in chosen_pids:
        if pid in pool and pid not in keep:
            items.append((pid, pool[pid]))
            keep.add(pid)

    names, pids, M = [], [], []
    for pid, (name, v) in items:
        pids.append(pid); names.append(name); M.append(v)
    M = np.vstack(M)

    pca = PCA(n_components=2, random_state=random_state)
    pca.fit(M if vec_syn is None else np.vstack([M, vec_syn]))
    XY = pca.transform(M)

    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=120)
    ax.scatter(XY[:, 0], XY[:, 1], s=14, alpha=0.55, label="Jugadores")

    if chosen_pids:
        x_ch, y_ch, labels_ch = [], [], []
        for pid in chosen_pids:
            if pid in pids:
                i = pids.index(pid)
                x_ch.append(XY[i, 0]); y_ch.append(XY[i, 1])
                labels_ch.append(names[i] if names[i] else pid)
        if x_ch:
            ax.scatter(x_ch, y_ch, s=90, marker="^", edgecolor="k", linewidths=0.8,
                       label="Elegidos")
            if annotate:
                for (x, y, lab) in zip(x_ch, y_ch, labels_ch):
                    ax.text(x, y, lab, fontsize=9, ha="left", va="bottom")

    if vec_syn is not None:
        syn_xy = pca.transform(vec_syn.reshape(1, -1))
        ax.scatter(syn_xy[0, 0], syn_xy[0, 1], s=180, marker="*",
                   edgecolor="k", linewidths=1.0, label="Perfil sintético")

    ax.legend()
    fig.tight_layout()
    return fig

# ======================================================
# UI principal
# ======================================================
def render_frankenstein():
    st.title("🧪 Frankenstein — Comparar 2 jugadores en Embeddings (Doc2Vec/Word2Vec)")

    # Mostrar info de cwd para depurar rutas (ayuda cuando “la ruta está bien” pero el proceso está en otro cwd)
    st.caption(f"Working dir: {Path.cwd().as_posix()}")
    default_docs, _ = _default_paths()

    # Input de ruta (en la propia página; nada en la barra lateral)
    path_docs = st.text_input(
        "Ruta al archivo player_docs_*.pkl.gz",
        value=default_docs.as_posix()
    )

    # Carga
    try:
        player_docs = load_player_docs(path_docs)
    except FileNotFoundError:
        parent = Path(path_docs).resolve().parent
        children = ", ".join([p.name for p in parent.glob("*")]) if parent.exists() else "(directorio no existe)"
        st.error(f"No encontré el archivo: {Path(path_docs).resolve().as_posix()}\n"
                 f"Directorio padre: {parent.as_posix()}\n"
                 f"Contenido del directorio: {children}")
        return
    except Exception as e:
        st.error(f"Error cargando player_docs: {e}")
        return

    st.success(f"Cargados {len(player_docs)} jugadores")
    # Entrena W2V rápido desde docs (o trae uno pre-entrenado si lo tienes)
    w2v = train_w2v_from_docs(player_docs)
    w2v_pool = build_w2v_pool(player_docs, w2v)
    sif_pool = build_w2v_sif(player_docs, w2v, a=1e-3, stop_top=35, remove_pc=True)

    # Opciones de selección (en la página)
    all_pids = [str(pid) for (pid, _) in player_docs.keys()]
    pid_to_name = {str(pid): name for (pid, name) in player_docs.keys()}

    colA, colB = st.columns(2)
    with colA:
        p1 = st.selectbox("Jugador 1", options=all_pids, format_func=lambda pid: f"{pid_to_name.get(pid,'?')} ({pid})")
    with colB:
        p2 = st.selectbox("Jugador 2", options=all_pids, index=min(1, len(all_pids)-1),
                          format_func=lambda pid: f"{pid_to_name.get(pid,'?')} ({pid})")

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        alpha = st.slider("Peso Jugador 1 (α)", 0.0, 1.0, 0.6, 0.05)
    with col2:
        beta = st.slider("Peso Jugador 2 (β)", 0.0, 1.0, 0.4, 0.05)
    with col3:
        space = st.radio("Espacio", ["sif", "w2v"], index=0, horizontal=True)

    # Botón para graficar
    if st.button("Generar comparación", use_container_width=True):
        pool = sif_pool if space == "sif" else w2v_pool
        vec_syn, usados = synthesize_embedding(
            components=[(p1, alpha), (p2, beta)],
            space=space, sif_pool=sif_pool, w2v_pool=w2v_pool, normalize=True
        )
        fig = plot_projection_2d_highlight(
            pool=pool, vec_syn=vec_syn, chosen_pids=[p1, p2], sample=600, annotate=True
        )
        st.pyplot(fig, use_container_width=True)