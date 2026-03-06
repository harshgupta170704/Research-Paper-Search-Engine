import streamlit as st
from sentence_transformers import SentenceTransformer
from endee import Endee

st.set_page_config(page_title="AI Research Paper Search", layout="wide")

st.title("AI Semantic Research Paper Search Engine")

st.write("Search research papers using semantic similarity.")

model = SentenceTransformer("all-MiniLM-L6-v2")

client = Endee()
index = client.get_index("papers")

query = st.text_input("Enter your research query")

if st.button("Search"):

    if query:

        vector = model.encode(query).tolist()

        results = index.query(vector=vector, top_k=5)

        st.subheader("Top Results")

        for r in results:

            st.markdown(f"**Title:** {r['meta']['title']}")
            st.markdown(f"Similarity Score: {r['similarity']}")
            st.markdown("---")
