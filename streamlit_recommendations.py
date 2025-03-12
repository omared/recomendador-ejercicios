import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Cargar modelo y embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = np.load("embeddings.npy")
data = np.load("data.npy", allow_pickle=True)

def get_recommendations(user_input, top_n=5):
    user_embedding = model.encode([user_input])
    similarities = np.dot(embeddings, user_embedding.T).flatten()
    best_indices = similarities.argsort()[-top_n * 2:][::-1]  # Duplicamos el top_n y filtramos luego

    categories = {
        "Ejercicios": [],
        "Alimentos": [],
        "Suplementos": [],
        "Hábitos saludables": [],
        "Rutinas de entrenamiento": [],
        "Técnicas de recuperación": [],
        "Planes de alimentación": []
    }

    for idx in best_indices:
        item = data[idx]
        for category in categories.keys():
            if category.lower() in item.lower():
                categories[category].append(item)

    # Seleccionar una recomendación de cada categoría (si hay disponibles)
    final_recommendations = []
    for recs in categories.values():
        if recs:
            final_recommendations.append(recs[0])

    return final_recommendations[:top_n]



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
