import streamlit as st
import pandas as pd
import numpy as np
import itertools
import pulp
import tensorflow as tf
from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
import time

# --- Configuración de la interfaz ---
st.set_page_config(page_title="🌌 Agrupador de Personalidades OCEAN", layout="wide", page_icon="🌠")

# --- CSS personalizado para mejorar estilo visual ---
st.markdown("""
    <style>
    .main { background: linear-gradient(145deg, #1e1e2f, #2b2b44); color: #fff; }
    .stButton button { background-color: #4b8bbe; color: white; border-radius: 10px; }
    .stDownloadButton button { background-color: #3aa6b9; color: white; border-radius: 10px; }
    .stDataFrame, .stTable { background-color: #ffffff10; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# --- Variables globales ---
MODEL_DIR = "model2_pandora"
MAX_LEN   = 70
LABELS    = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]

LABEL_DESCRIPTIONS = {
    "Openness": "🧠 **Openness**: Apertura a nuevas experiencias, creatividad, curiosidad intelectual y sensibilidad estética.",
    "Conscientiousness": "📘 **Conscientiousness**: Organización, disciplina, responsabilidad y orientación a logros.",
    "Extraversion": "📣 **Extraversion**: Tendencia a buscar la compañía de otros, energía social, entusiasmo.",
    "Agreeableness": "💞 **Agreeableness**: Amabilidad, cooperación, empatía y altruismo.",
    "Neuroticism": "🌪️ **Neuroticism**: Tendencia a experimentar emociones negativas como ansiedad, ira o depresión."
}

# --- Cargar modelo/tokenizador ---
@st.cache_resource
def load_model():
    with st.spinner("🔍 Cargando modelo de personalidad..."):
        time.sleep(1)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_DIR, local_files_only=True)
    return tokenizer, model

tokenizer, model = load_model()

# --- Funciones ---
def predict(text):
    inp = tokenizer(text, truncation=True, padding=True,
                    max_length=MAX_LEN, return_tensors="tf")
    probs = tf.sigmoid(model(inp)[0])[0].numpy()
    return np.round(probs * 100, 1)

@st.cache_data
def calcular_matrices():
    percentile_bins = [">=90%","80-89%","70-79%","60-69%","50-59%",
                       "40-49%","30-39%","20-29%","10-19%","<10%"]
    def m(vals): return pd.DataFrame(vals, index=percentile_bins, columns=percentile_bins)
    return {
        "Extraversion": m([[3,3,3,3,2,2,1,1,0,0],[3,3,3,3,2,2,1,1,1,0],[3,3,3,3,2,2,1,1,1,1],[3,3,3,3,2,2,2,1,1,1],[2,2,2,2,3,2,2,1,1,1],[2,2,2,2,2,3,2,2,1,1],[1,1,1,2,2,2,3,2,2,1],[1,1,1,1,1,2,2,3,2,2],[0,1,1,1,1,1,2,2,3,2],[0,0,1,1,1,1,1,2,2,3]]),
        "Openness": m([[3,3,2,2,1,1,0,0,0,0],[3,3,2,2,1,1,0,0,0,0],[2,2,3,2,2,1,1,0,0,0],[2,2,2,3,2,2,1,1,0,0],[1,1,2,2,3,2,2,1,1,0],[1,1,1,2,2,3,2,2,1,1],[0,0,1,1,2,2,3,2,2,1],[0,0,0,1,1,2,2,3,2,2],[0,0,0,0,1,1,2,2,3,2],[0,0,0,0,0,1,1,2,2,3]]),
        "Conscientiousness": m([[3,3,2,2,1,0,0,0,0,0],[3,3,2,3,2,1,0,0,0,0],[2,2,3,2,3,2,1,0,0,0],[2,3,2,3,2,2,2,1,0,0],[1,2,3,2,3,2,2,2,1,0],[0,1,2,2,2,3,2,2,2,1],[0,0,1,2,2,2,3,2,2,1],[0,0,0,1,2,2,2,3,2,2],[0,0,0,0,1,2,2,2,3,2],[0,0,0,0,0,1,1,2,2,3]]),
        "Agreeableness": m([[3,2,2,2,1,1,1,0,0,0],[2,3,2,2,2,1,1,0,0,0],[2,2,3,2,2,2,1,1,0,0],[2,2,2,3,2,2,2,1,1,0],[1,2,2,2,3,2,2,2,2,1],[1,1,2,2,2,3,2,2,2,2],[1,1,1,2,2,2,3,2,2,2],[0,0,1,2,2,2,2,3,2,2],[0,0,0,1,2,2,2,2,3,2],[0,0,0,0,1,2,2,2,2,3]]),
        "Neuroticism": m([[1,1,2,2,2,1,0,0,0,0],[1,1,2,2,2,1,1,0,0,0],[2,2,1,2,2,2,1,1,0,0],[2,2,2,3,2,2,2,1,1,0],[2,2,2,2,3,2,2,2,2,1],[1,1,2,2,2,3,2,2,2,2],[0,1,1,2,2,2,3,2,2,2],[0,0,1,1,2,2,2,3,2,2],[0,0,0,1,2,2,2,2,3,2],[0,0,0,0,1,2,2,2,2,3]])
    }

compat_matrices = calcular_matrices()
percentile_bins = list(compat_matrices["Extraversion"].index)

def decile(score:int):
    for thr,label in zip([90,80,70,60,50,40,30,20,10,0], percentile_bins):
        if score >= thr: return label

def compat(u1, u2):
    total = 0
    for trait, matrix in compat_matrices.items():
        total += matrix.loc[decile(u1[trait]), decile(u2[trait])]
    return total

# --- Interfaz principal ---
st.title("🌠 Agrupador de Personalidades (Modelo Big Five)")
file = st.file_uploader("📤 Sube el archivo `new_users.csv` (20 filas, columnas id,text)", type="csv")

if file:
    df_users = pd.read_csv(file)
    assert df_users.shape[0] == 20, "⚠️ Debe haber exactamente 20 usuarios"

    with st.spinner("🧠 Analizando personalidades..."):
        scores = np.vstack(df_users["text"].apply(predict).values)
    df_scores = pd.DataFrame(scores, columns=LABELS)
    df_all = pd.concat([df_users["id"], df_scores], axis=1).set_index("id")

    st.success("✅ Análisis completado.")
    st.subheader("📊 Perfiles OCEAN")
    st.dataframe(df_all)

    with st.expander("🔎 ¿Qué mide cada dimensión del modelo OCEAN?"):
        for label in LABELS:
            st.markdown(LABEL_DESCRIPTIONS[label])

    # Compatibilidad entre todos
    ids = df_all.index.tolist()
    pairs = {frozenset({a,b}): compat(df_all.loc[a], df_all.loc[b])
             for a,b in itertools.combinations(ids,2)}

    groups = []
    for size in (3,4):
        for combo in itertools.combinations(ids, size):
            score = sum(pairs[frozenset({x,y})] for x,y in itertools.combinations(combo,2))
            groups.append((combo, score))

    # ILP para agrupamiento
    prob = pulp.LpProblem("GroupOpt", pulp.LpMaximize)
    y = {i: pulp.LpVariable(f"g{i}", 0, 1, cat="Binary") for i in range(len(groups))}
    prob += pulp.lpSum(s * y[i] for i,(_,s) in enumerate(groups))
    for u in ids:
        prob += pulp.lpSum(y[i] for i,(g,_) in enumerate(groups) if u in g) == 1
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    chosen = [groups[i] for i in y if pulp.value(y[i]) == 1]

    st.subheader("🤝 Grupos Óptimos")
    for grp, score in chosen:
        st.markdown(f"✅ **Grupo:** {', '.join(grp)} → Compatibilidad total: `{score}`")

    out = []
    for grp,_ in chosen:
        gname = "+".join(grp)
        for u in grp:
            out.append({"Usuario": u, "Grupo": gname})
    df_asign = pd.DataFrame(out)

    st.download_button("📥 Descargar agrupamiento CSV", df_asign.to_csv(index=False).encode(), "asignaciones_grupos.csv")