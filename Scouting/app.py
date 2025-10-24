# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gensim.models import Doc2Vec

from src.events_embeddings.embeddings import doc2vec_neighbors
from src.shared.helpers import radar_compare_players, pick_player_row, plot_embedding_scatter
from src.fingerprints.prep import (
    concat_player_season_stats, filter_by_minutes,
    attach_positions, drop_goalkeepers,
)
from src.fingerprints.pca_select import (
    build_fingerprint_df_autoK_FROM_DF,
)
from src.fingerprints.clustering import similares


# =========================
# CONFIG UI
# =========================
st.set_page_config(page_title="Scouting Workspace", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    .smallcaps { font-variant: small-caps; letter-spacing: .5px; }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# HELPERS: CARGA DE DATOS/MODELOS
# =========================
@st.cache_data(show_spinner=True)
def load_season_stats_parquet():
    SEASONS = [(317, "2024/2025"), (281, "2023/2024"), (235, "2022/2023"), (108, "2021/2022")]
    frames = []
    for sid, sname in SEASONS:
        p = f"data/Jugadores/player_season_stats_{sid}.parquet"
        df = pd.read_parquet(p, engine="pyarrow")
        if "season_id" not in df.columns:
            df["season_id"] = sid
        if "season_name" not in df.columns:
            df["season_name"] = sname
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

@st.cache_data(show_spinner=True)
def load_positions_csv():
    return pd.read_csv("data/tpi_player_season_ligamx.csv", usecols=["season_id", "player_id", "position"])

@st.cache_data(show_spinner=True)
def build_Zall(stats: pd.DataFrame, pos_csv: pd.DataFrame, min_minutes=900):
    all_stats = concat_player_season_stats({
        (int(sid), str(sid)): stats[stats["season_id"] == sid]
        for sid in stats["season_id"].dropna().unique()
    })
    all_stats = attach_positions(all_stats, "data/tpi_player_season_ligamx.csv")
    all_stats = drop_goalkeepers(all_stats)
    all_stats = filter_by_minutes(all_stats, min_minutes=min_minutes)
    Zall, pca, scalers, kinfo = build_fingerprint_df_autoK_FROM_DF(
        all_stats,
        k_method="parallel",
        variance_threshold=0.85,
        group_standardization=("season_id",),
        random_state=42
    )
    return Zall

@st.cache_resource(show_spinner=True)
def load_d2v_model():
    return Doc2Vec.load("data/Modelos/doc2vec_128d_20251023-145024.model")


# =========================
# SIDEBAR NAVIGATION
# =========================
st.sidebar.title("Scouting")
section = st.sidebar.radio("Ir a:", ["Inicio", "Análisis", "Futuro"], index=0)
subpage = None
if section == "Análisis":
    subpage = st.sidebar.radio("Módulo:", ["Cluster", "Embeddings"], index=0)


# =========================
# HOME
# =========================
if section == "Inicio":
    st.title("Inicio")
    st.markdown("""
    ## ¿Quiénes somos? 

    #### Carlos Lugo: 
                
    Tengo 21 años y estoy formándome como Ingeniero en Computación en el ITAM. Aunque mi carrera abarca muchos temas, mi verdadera pasión es la Ciencia de Datos. Estoy convencido de su potencial y, por ello, dedico parte de mi tiempo a realizar investigación en el INMEGEN, donde puedo aplicar modelos analíticos a datos del mundo real.
                
    #### Emilio González:
                 
    Tengo 21 años y estoy concluyendo la carrera de Ingeniería en Computación. Lo que más me interesa es la IA y la ciencia de datos, de igual forma estoy realizando una investigación en el INMEGEN y soy apasionado al deporte 
                           
    #### Alexa Morales:
                
    Soy Alexa Morales, estudiante de Ingeniería en Computación e Ingeniería en Mecatrónica en el ITAM. Me apasiona crear soluciones que combinen tecnología, diseño e impacto real. He trabajado en proyectos de innovación dentro del Makers Lab del ITAM, donde lidero el área de diseño, y también en análisis de datos médicos en el Instituto Nacional de Medicina Genómica.
    Me encanta aprender, experimentar con nuevas ideas y trabajar en equipo para convertir conceptos en proyectos tangibles, desde código hasta prototipos impresos en 3D. Creo que la ingeniería es una forma de cambiar el mundo, una línea de código (o una pieza impresa) a la vez.

    #### Regina Sierra:
                       
    Tengo 19 años y estudio Administración, con un especial interés en la mercadotecnia. Me apasiona el deporte y disfruto aprender sobre él desde distintas perspectivas. También me gusta leer, escribir, aprender nuevos idiomas y editar videos. Me considero una persona perfeccionista y comprometida, que disfruta los retos: desde correr medios maratones, hasta entrar a este equipo en un tema en el que no soy experta.
                      
    """)
    st.stop()


# =========================
# LOAD DATA
# =========================
with st.spinner("Cargando datos…"):
    stats = load_season_stats_parquet()
    pos_csv = load_positions_csv()
    Zall = build_Zall(stats, pos_csv, min_minutes=900)

# =========================
# CLUSTER
# =========================
if section == "Análisis" and subpage == "Cluster":
    all_seasons = sorted(Zall["season_id"].dropna().unique().astype(int))
    season_id = st.sidebar.selectbox("Temporada:", options=["(todas)"] + list(map(int, all_seasons)), index=0)

    df_filter = Zall.copy()
    if season_id != "(todas)":
        df_filter = df_filter[df_filter["season_id"] == int(season_id)]

    players_df = (
        df_filter[["player_id", "player_name"]]
        .dropna().drop_duplicates().sort_values("player_name")
    )
    player_name_options = players_df["player_name"].astype(str).tolist()
    player_name_pick = st.sidebar.selectbox("Jugador A:", options=player_name_options, index=0)
    base_pid = int(players_df.loc[players_df["player_name"] == player_name_pick, "player_id"].iloc[0])

    st.title("Cluster")
    colL, colR = st.columns([1.3, 1])

    with colL:
        st.subheader(f"Jugador: {player_name_pick}")
        rowA = pick_player_row(Zall, base_pid, None if season_id == "(todas)" else int(season_id))
        if rowA is not None:
            pname = str(rowA.get("player_name", player_name_pick))
            team = rowA.get("team_name", "—")
            sname = rowA.get("season_name", "—")
            st.markdown(f"*Equipo:* {team}")
            st.markdown(f"*Temporada:* {sname}")

        sim_df = similares(Zall, player_id=base_pid, k=20).reset_index(drop=True)
        
        sim_df = sim_df.drop_duplicates(subset=['player_id', 'season_id'], keep='first')
        sim_df.index = np.arange(1, len(sim_df) + 1)
        sim_df = sim_df.head(10).copy()

        st.markdown("#### Jugadores similares:")
        st.dataframe(sim_df[["player_name", "team_name", "season_name"]], use_container_width=True, height=360)

    with colR:
        comp_options = sim_df["player_name"].astype(str).tolist()
        comp_pick = st.selectbox("Comparar con:", ["(ninguno)"] + comp_options, index=0)
        comp_pid = base_pid if comp_pick == "(ninguno)" else int(sim_df.loc[sim_df["player_name"] == comp_pick, "player_id"].iloc[0])
        fig = radar_compare_players(
            Zall, base_pid, comp_pid,
            season_id=None if season_id == "(todas)" else int(season_id),
            scale_mode="dataset"
        )
        st.pyplot(fig, use_container_width=True)


# =========================
# EMBEDDINGS
# =========================
if section == "Análisis" and subpage == "Embeddings":
    st.title("Embeddings")
    d2v = load_d2v_model()

    # --- Selector SOLO de jugador (sin temporada) ---
    players_df = (
        Zall[["player_id", "player_name", "team_name"]]
        .dropna().drop_duplicates().sort_values("player_name")
    )
    player_name_options = players_df["player_name"].astype(str).tolist()
    player_name_pick = st.sidebar.selectbox("Jugador A:", options=player_name_options, index=0)

    base_pid = int(
        players_df.loc[players_df["player_name"] == player_name_pick, "player_id"].iloc[0]
    )
    base_team = str(
        players_df.loc[players_df["player_name"] == player_name_pick, "team_name"].iloc[0]
    )

    # --- Vecinos Doc2Vec del jugador base ---
    try:
        neigh_df = doc2vec_neighbors(float(base_pid), d2v, k=200)
    except Exception as e:
        st.error(f"No pude obtener vecinos Doc2Vec para {player_name_pick}: {e}")
        st.stop()

    # Completar nombre/equipo a partir de Zall
    pid2name = dict(zip(Zall["player_id"], Zall["player_name"]))
    pid2team = dict(zip(Zall["player_id"], Zall["team_name"]))
    neigh_df["player_name"] = neigh_df["player_id"].apply(
        lambda x: pid2name.get(x, f"player_{int(float(x))}")
    )
    neigh_df["team_name"] = neigh_df["player_id"].apply(lambda x: pid2team.get(x, "—"))

    # Eliminar filas con nombres tipo "player_####"
    mask_fake = neigh_df["player_name"].astype(str).str.match(r"player_\d+", case=False)
    neigh_df = neigh_df[~mask_fake].copy()

    # Top (solo con nombre/equipo, sin "similaridad")
    top_tbl = neigh_df[["player_id", "player_name", "team_name"]].copy()
    top_tbl.index = np.arange(1, len(top_tbl) + 1)

    colA, colB = st.columns([1.1, 1])

    with colA:
        st.subheader(f"Jugador: {player_name_pick}")
        st.caption(f"Equipo: {base_team or '—'}")

        st.markdown("#### Vecinos (Doc2Vec)")
        st.dataframe(
            top_tbl[["player_name", "team_name"]],
            use_container_width=True,
            height=360
        )

       
    with colB:
        comp_options = top_tbl["player_name"].astype(str).tolist()
        comp_pick = st.selectbox(
            "Comparar en el mapa:", ["(ninguno)"] + comp_options, index=0, key="emb_comp"
        )
        comp_pid = None
        comp_team = "—"
        if comp_pick != "(ninguno)":
            row_comp = top_tbl.loc[top_tbl["player_name"] == comp_pick].iloc[0]
            comp_pid = float(row_comp["player_id"])
            comp_team = str(row_comp["team_name"])

        if comp_pid is not None:
            st.caption(f"Comparando con: *{comp_pick}* — {comp_team}")


        # Diccionarios para anotar nombres en el gráfico
        pid2name_all = dict(zip(Zall["player_id"], Zall["player_name"]))

        # Resaltar jugador base y opcionalmente el comparador
        highlight = (
            (float(base_pid), float(comp_pid))
            if comp_pid is not None
            else (float(base_pid), float(base_pid))
        )

        try:
            fig, ax, coords = plot_embedding_scatter(
                d2v,
                highlight_pids=highlight,
                pid2name=pid2name_all,
                subset_pids=None,       # podrías pasar un subconjunto si quieres
                figsize=(7, 5),
                point_size=14,
                alpha=0.4,
                random_state=42
            )
            st.pyplot(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"No se pudo generar la proyección PCA 2D: {e}")




# =========================
# FUTURO
# =========================
if section == "Futuro":
    st.title("Futuro")
    st.markdown("""
    Próximos módulos:
    Estamos convencidos de que nuestro proyecto es revolucionario e innovador, pues proporciona una base muy sólida con muchas posibles aplicaciones. 
    Primero, podría alcanzar un nivel de detalle mayor. Por ejemplo, si transformamos la palabra “pase” por otras más, como “pase progresivo/ elevado/ con la izquierda”, enriquecemos los embeddings y podemos refinar el análisis. 
    Segundo, análisis de estilos opuestos, con vectores ortogonales. Esto quizás no tenga uso en scouting, pero sí sirve para hacer un análisis táctico. Por ejemplo, si buscamos el jugador más ortogonal a un delantero, podemos entender qué tipo de defensor podría “neutralizarlo” con mayor eficacia. Así podrían los entrenadores anticipar estrategias para contrarrestar a buenos jugadores enemigos. 
    Tercero, como combinamos jugadores, en un futuro si seguimos entrenando esto, vamos a poder hacer opciones de tipo Kevin Álvarez - pases al centro + tiro= un jugador más cercano a esas características.
    """)