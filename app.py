import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
model_dir = '/content/drive/MyDrive/ColabNotebooks/Models/best_fine_tuned_bert'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Apply custom styling for the title (centered)
st.markdown(
    """
    <h3 style="text-align: center;">Sentiment Analysis on Airline Reviews</h3>
    <hr style="width:50%; margin:auto;">
    """,
    unsafe_allow_html=True
)

# Add an image with reduced width and align it to the right, replace deprecated 'use_column_width' with 'use_container_width'
st.image("/content/drive/MyDrive/Colab Notebooks/Sentiment Analysis cover pic.jpg", width=400, use_container_width=True)

# Make the text bold and set the same size as "History" section, remove the ** from "How was your experience"
st.markdown("<h4 style='text-align: center;'>How was your experience?</h4>", unsafe_allow_html=True)

# User input text box
user_input = st.text_area("Enter your review here:")

# Sentiment Analysis function
def sentiment_analyzer(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).item()
    return predictions

# Customize the "Analyze Sentiment" button with dark blue color, center it
button_style = """
    <style>
        .stButton>button {
            background-color: #003366;  /* Dark Blue */
            color: white;
            font-size: 16px;
            border-radius: 5px;
            width: 100%;  /* Ensure it stretches across */
            display: block;
            margin: 0 auto;  /* Center button */
        }
    </style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

# When the user clicks the button, analyze sentiment
if st.button("Analyze Sentiment"):
    if user_input:
        sentiment = sentiment_analyzer(user_input)
        sentiment_label = "Positive 😊" if sentiment == 1 else "Negative 😞"

        # Conditional color for prediction box with lighter shades
        if sentiment == 1:
            prediction_color = "#66cc66"  # Light Green
        else:
            prediction_color = "#ff6666"  # Light Red

        # Display the result with a background color based on sentiment
        st.markdown(f"""
            <div style="background-color:{prediction_color}; padding: 10px; border-radius: 5px; color: white; text-align: center;">
                <h4>Prediction: {sentiment_label}</h4>
            </div>
        """, unsafe_allow_html=True)

        # Add to history
        st.session_state.history.append({
            "Review": user_input,
            "Sentiment": sentiment_label
        })
    else:
        st.warning("Please enter a review to analyze.")

# Display the history of reviews and predictions
if st.session_state.history:
    st.subheader("History")
    # Create a dataframe to display the history
    import pandas as pd
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df)
