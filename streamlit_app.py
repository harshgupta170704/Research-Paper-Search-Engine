import streamlit as st
from sentence_transformers import SentenceTransformer
from endee import Endee

st.title("AI Semantic Document Search")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Endee
client = Endee()
index = client.get_index("papers")

query = st.text_input("Enter your search query")

if st.button("Search"):

    if query:
        vector = model.encode(query).tolist()

        results = index.query(vector=vector, top_k=5)

        st.subheader("Results")

        for r in results:
            st.write("Title:", r["meta"]["title"])
            st.write("Similarity:", r["similarity"])
            st.write("---")
