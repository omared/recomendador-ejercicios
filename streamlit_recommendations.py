import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar modelo y embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = np.load("embeddings.npy")
data = np.load("data.npy", allow_pickle=True)

def obtener_recomendaciones(input_text, top_n=5):
    input_embedding = model.encode([input_text])
    similitudes = cosine_similarity(input_embedding, embeddings)[0]
    indices_top = np.argsort(similitudes)[::-1][:top_n]
    return [data[i] for i in indices_top]

# Interfaz con Streamlit
st.title("Recomendador de Ejercicios y Alimentación")

# Inputs del usuario
edad = st.number_input("Edad", min_value=10, max_value=100, value=25)
peso = st.number_input("Peso (kg)", min_value=30, max_value=200, value=70)
genero = st.selectbox("Género", ["Masculino", "Femenino", "Otro"])
objetivo = st.selectbox("Objetivo", ["Ganar masa muscular", "Perder peso", "Mantenerse saludable"])

if st.button("Obtener Recomendaciones"):
    consulta = f"{objetivo} para una persona de {edad} años y {peso} kg"
    recomendaciones = obtener_recomendaciones(consulta)
    st.subheader("Recomendaciones:")
    for rec in recomendaciones:
        st.write(f"- {rec}")
