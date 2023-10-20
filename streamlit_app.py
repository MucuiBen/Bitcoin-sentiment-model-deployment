import subprocess

# Install required dependencies from requirements.txt
subprocess.call(['pip', 'install', '-r', 'requirements.txt'])

import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import logging
import os

# Set page configuration (must be called first)
st.set_page_config(
    page_title="Bitcoin Sentiment Analysis App",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

# Logging setup
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

# Set OpenMP environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Deprecated st.cache replaced with st.cache_data
@st.cache(allow_output_mutation=True)
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenizer.add_special_tokens({'additional_special_tokens': ['[SENTIMENT]', '[CATEGORY]']})
    model_path = "final-model"
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

model, tokenizer = load_model()

# Streamlit UI (Title and Input)
st.title('Bitcoin Sentiment Analysis App')

# Add theme customization with new color values
primaryColor = '#FF5733'  # Bright Orange
backgroundColor = '#1F1F1F'  # Dark Gray
secondaryBackgroundColor = '#FFD700'  # Gold

st.markdown(
    f"""
    <style>
        body {{
            font-family: Roboto, sans-serif;
            color: #333;
            background-color: {backgroundColor};
        }}

        .st-bw {{
            padding: 15px;
            margin: 20px 0;
            background-color: #fff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }}

        .stButton > button {{
            background-color: {primaryColor};
        }}

        .css-17iud4w-buttonGroupContainer {{
            background-color: {secondaryBackgroundColor};
        }}

        .st-bn {{
            color: #fff;
            background-color: {primaryColor};
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Input text area
text = st.text_area("Enter Text for Analysis:")

# Prediction
if st.button('Analyze'):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    _, prediction = torch.max(probs, dim=1)

    # Display sentiment prediction
    sentiment_map = {0: "Negative 😞", 1: "Neutral 😐", 2: "Positive 😃"}
    sentiment = sentiment_map[prediction.item()]
    st.markdown(f'<p style="font-size: 20px; color: {"red" if prediction.item() == 0 else "green" if prediction.item() == 2 else "black"};">Sentiment: {sentiment}</p>', unsafe_allow_html=True)

    # Log user interaction
    logging.info(f"User Input: {text} | Model Prediction: {sentiment}")

    # Sentiment Distribution Pie Chart and Word Cloud
    st.subheader("Sentiment Distribution & Word Cloud")

    # Create two columns for pie chart and word cloud
    col1, col2 = st.columns(2)

    # Pie chart in the first column
    with col1:
        sentiment_labels = ['Negative', 'Neutral', 'Positive']
        probs_data = probs[0].detach().numpy()

        fig, ax1 = plt.subplots()

        wedges, texts = ax1.pie(probs_data, startangle=90)
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        ax1.legend(wedges, sentiment_labels, title="Sentiments", loc="best")
        ax1.axis('equal')

        st.pyplot(fig)

        st.write(f"Negative sentiment: {probs_data[0]:.2f}%")
        st.write(f"Neutral sentiment: {probs_data[1]:.2f}%")
        st.write(f"Positive sentiment: {probs_data[2]:.2f}%")

    # Word Cloud in the second column
    with col2:
        # Use the entire text for generating Word Cloud
        wordcloud_all = WordCloud(width=800, height=400, background_color='white', stopwords=set(['the', 'and', 'to'])).generate(text)
        
        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(wordcloud_all, interpolation='bilinear')
        ax.axis('off')
        
        # Display the Matplotlib figure using st.pyplot()
        st.pyplot(fig)

# Sidebar Info
st.sidebar.title("About App")
st.sidebar.info("This is a sentiment analysis app using a fine-tuned DistilBERT model.")
st.sidebar.title("Model Predictions")
st.sidebar.write("0: Negative, 1: Neutral, 2: Positive")

# Additional Sidebar Features
st.sidebar.subheader("Model Details")
st.sidebar.write("Model Type: DistilBERT")
st.sidebar.write("Training Data: Bitcoin Reddit Comments")
st.sidebar.write("Accuracy: 91% (Train), 88% (Validation)")
st.sidebar.markdown("[More about BERT and its variants](https://arxiv.org/abs/1810.04805)")
st.sidebar.markdown("[Some BERT illustrations](https://jalammar.github.io/illustrated-bert/)")

# Sidebar Info and Feedback Form
st.sidebar.subheader("Feedback")
feedback = st.sidebar.text_area("Please leave your feedback here:")
if st.sidebar.button("Submit Feedback"):
    logging.info(f"User Feedback: {feedback}")
    st.sidebar.write("Thank you for your feedback!")
