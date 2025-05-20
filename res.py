import streamlit as st
from transformers import TFAutoModelForSeq2SeqLM, TFAutoModelForQuestionAnswering, AutoTokenizer
import tensorflow as tf

# Set page config
st.set_page_config(page_title="AI Text Apps", layout="wide")

# Custom CSS to make the app beautiful and interactive
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff;
            font-family: 'Roboto', sans-serif;
        }
        .title {
            text-align: center;
            color: #1e90ff;
            font-size: 40px;
            font-weight: bold;
        }
        .subheader {
            color: #ff6347;
            font-size: 22px;
            font-weight: 500;
        }
        .stTextArea, .stTextInput {
            border: 2px solid #1e90ff;
            border-radius: 8px;
            padding: 10px;
        }
        .stButton>button {
            background-color: #1e90ff;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s;
        }
        .stButton>button:hover {
            background-color: #4682b4;
        }
        .stSelectbox select {
            font-size: 16px;
            font-weight: 500;
            border-radius: 8px;
        }
        .stWarning {
            background-color: #fff3cd;
            color: #856404;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }
        .stSuccess {
            background-color: #d4edda;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin-top: 10px;
        }
        .stSpinner {
            display: block;
            margin-top: 20px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Load Summarization Model
@st.cache_resource
def load_summary_model():
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Load Question Answering Model
@st.cache_resource
def load_qa_model():
    model_name = "distilbert-base-uncased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

# Load both models
summary_tokenizer, summary_model = load_summary_model()
qa_tokenizer, qa_model = load_qa_model()

# Summarization Function
def generate_summary(text):
    input_text = "summarize: " + text
    inputs = summary_tokenizer(input_text, return_tensors="tf", max_length=512, truncation=True)
    summary_ids = summary_model.generate(
        inputs['input_ids'],
        max_length=50,
        min_length=10,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Question Answering Function
def generate_answer(context, question):
    inputs = qa_tokenizer.encode_plus(question, context, return_tensors="tf")
    input_ids = inputs["input_ids"].numpy()[0]

    outputs = qa_model(inputs)
    start = tf.argmax(outputs.start_logits, axis=1).numpy()[0]
    end = tf.argmax(outputs.end_logits, axis=1).numpy()[0] + 1

    answer = qa_tokenizer.convert_tokens_to_string(
        qa_tokenizer.convert_ids_to_tokens(input_ids[start:end])
    )
    return answer.strip()

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #1e90ff;'>🧠 **Text Summarizer & QA App**</h1>", unsafe_allow_html=True)

# Option to select the app
app_mode = st.selectbox("🔄 Select an option", ["Text Summarizer", "Question Answering"])

if app_mode == "Text Summarizer":
    st.markdown("### 📝 Enter a paragraph and get a summary .")
    
    # Input for summarization
    text_input = st.text_area("📄 **Enter Paragraph**", height=200, placeholder="Paste your paragraph here...", key="summarize", label_visibility="visible")
    
    if st.button("🔍 Generate Summary"):
        if text_input.strip():
            with st.spinner("Generating summary..."):
                summary = generate_summary(text_input)
            st.subheader("🧾 **Summary:**")
            st.success(summary)
        else:
            st.warning("Please enter some text to summarize.")

elif app_mode == "Question Answering":
    st.markdown("### ❓ Ask any question based on the given context.")
    
    # Input for question answering
    context = st.text_area("📘 **Enter Context**", height=200, placeholder="Paste your context here...", label_visibility="visible")
    question = st.text_input("❓ **Enter Your Question**", placeholder="What is the answer?", label_visibility="visible")
    
    if st.button("Get Answer"):
        if context.strip() == "" or question.strip() == "":
            st.warning("Please enter both context and question.")
        else:
            with st.spinner("Thinking... 🤔"):
                answer = generate_answer(context, question)
            st.success("✅ **Answer:**")
            st.markdown(f"**{answer}**")
