import streamlit as st
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import tensorflow as tf

# Load model and tokenizer once
@st.cache_resource
def load_model():
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Summary generation function
def generate_summary(text):
    input_text = "summarize: " + text
    inputs = tokenizer(input_text, return_tensors="tf", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=50,
        min_length=10,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit UI
st.title("🧠 Text Summarizer (T5-small)")
st.markdown("Enter a paragraph and get a summary using TensorFlow T5-small model.")

# Input area
text_input = st.text_area("📄 Enter Paragraph", height=200)

# Generate summary
if st.button("🔍 Generate Summary"):
    if text_input.strip():
        with st.spinner("Generating summary..."):
            summary = generate_summary(text_input)
        st.subheader("🧾 Summary:")
        st.success(summary)
    else:
        st.warning("Please enter some text to summarize.")
