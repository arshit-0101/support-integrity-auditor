import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title='Support Integrity Auditor', page_icon='🔍', layout='wide')

@st.cache_resource
def load_models():
    device = torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained('arshit0101/support-integrity-auditor', use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained('arshit0101/support-integrity-auditor', torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.eval()
    embedder = SentenceTransformer('paraphrase-MiniLM-L3-v2', device='cpu')
    return tokenizer, model, embedder, device

with st.spinner('Loading model...'):
    tokenizer, model, embedder, device = load_models()

st.success('Model loaded!')
st.title('Support Integrity Auditor')
st.write('App is running!')
