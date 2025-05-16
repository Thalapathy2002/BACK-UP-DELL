import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load SHL catalog
df = pd.read_csv('assessments.csv')

# Precompute embeddings
df['embedding'] = df['description'].apply(lambda x: model.encode(x, convert_to_tensor=True))

# Function to fetch text from URL
def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text()
    except Exception as e:
        return ""

# UI
st.title("🔍 SHL Test Recommender (RAG Tool)")
st.write("Enter a job description or URL to get matching SHL assessments.")

input_mode = st.radio("Input Mode", ["Text", "URL"])

if input_mode == "Text":
    user_input = st.text_area("Paste job description:", height=200)
elif input_mode == "URL":
    url_input = st.text_input("Paste URL:")
    if url_input:
        user_input = fetch_text_from_url(url_input)
    else:
        user_input = ""

if user_input:
    query_embedding = model.encode(user_input, convert_to_tensor=True)
    df['score'] = df['embedding'].apply(lambda x: float(util.cos_sim(query_embedding, x)))
    top_matches = df.sort_values('score', ascending=False).head(10)

    st.subheader("🔗 Recommended SHL Assessments")
    st.write("Based on similarity with your job description.")

    def yesno(val):
        return "✅" if val == "Yes" else "❌"

    result_df = pd.DataFrame({
        "Assessment Name": top_matches['name'],
        "URL": top_matches['url'].apply(lambda x: f"[Link]({x})"),
        "Remote Testing Support": top_matches['remote_support'].apply(yesno),
        "Adaptive/IRT": top_matches['adaptive_irt'].apply(yesno),
        "Duration": top_matches['duration'],
        "Test Type": top_matches['test_type']
    })

    st.write(result_df.to_markdown(index=False), unsafe_allow_html=True)
