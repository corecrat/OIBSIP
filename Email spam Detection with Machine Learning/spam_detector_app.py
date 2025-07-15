import streamlit as st
import pickle
import string
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.stem import PorterStemmer
from wordcloud import WordCloud

# Page configuration
st.set_page_config(page_title="Email Spam Detector", layout="centered")

# Custom UI styling
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }
    .stTextArea textarea {
        font-size: 1.1rem;
    }
    .stButton>button {
        background-color: #6c63ff;
        color: white;
        border: none;
        padding: 0.5em 1em;
        font-size: 1rem;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #574fd6;
    }
    .result {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 0.8rem;
        border-radius: 12px;
        background-color: #fff;
        box-shadow: 0 0 15px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Dark mode toggle
dark_mode = st.toggle("Dark Mode", value=False)
if dark_mode:
    st.markdown("""
        <style>
        body {
            background: linear-gradient(to right, #232526, #414345);
            color: white;
        }
        .result { background-color: #2a2b2e; color: white; }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body {
            background: linear-gradient(to right, #f5f7fa, #c3cfe2);
        }
        </style>
    """, unsafe_allow_html=True)

# Stopword set
stopwords = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
}

stemmer = PorterStemmer()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords]
    return " ".join(words)

# Load trained model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ✅ Step 6: Save the model and vectorizer as .pkl files
with open("spam_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully.")

# App title and input field
st.title("Email Spam Detector")
st.write("Enter a message to classify it as spam or not spam.")
user_input = st.text_area("Enter your message:", height=150)

# Store prediction history in session
if "log" not in st.session_state:
    st.session_state.log = []

# Prediction trigger
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        # Preprocess and predict
        cleaned_text = preprocess_text(user_input)
        vectorized_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        confidence = model.predict_proba(vectorized_text)[0][prediction] * 100

        # Format and display result
        label = "Spam" if prediction == 1 else "Ham"
        color = "red" if prediction == 1 else "green"
        st.markdown(
            f'<div class="result" style="color:{color};">Prediction: {label}<br>Confidence: {confidence:.2f}%</div>',
            unsafe_allow_html=True
        )

        # Save result to log
        st.session_state.log.append((user_input, label, f"{confidence:.2f}%"))

# Show prediction history
if st.toggle("Show Prediction History"):
    for msg, label, conf in reversed(st.session_state.log):
        st.markdown(f"**Message**: _{msg}_\n\n→ **{label}** with **{conf}** confidence\n---")

# Show bar chart of predictions
if st.toggle("Show Bar Chart"):
    if st.session_state.log:
        results = [label for _, label, _ in st.session_state.log]
        summary_df = pd.DataFrame(results, columns=["Prediction"])
        fig, ax = plt.subplots()
        summary_df["Prediction"].value_counts().plot(kind="bar", color=["green", "red"], ax=ax)
        ax.set_title("Prediction Counts")
        ax.set_xlabel("Message Type")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)
    else:
        st.info("No predictions yet.")

# Show pie chart of prediction distribution
if st.toggle("Show Pie Chart"):
    if st.session_state.log:
        pie_labels = [label for _, label, _ in st.session_state.log]
        pie_data = pd.Series(pie_labels).value_counts()
        fig2, ax2 = plt.subplots()
        ax2.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', colors=["green", "red"], startangle=90)
        ax2.axis("equal")
        ax2.set_title("Spam vs Ham Ratio")
        st.pyplot(fig2)
    else:
        st.info("No predictions to show.")

#Show word clouds for spam and ham messages
if st.toggle("Show WordCloud by Category"):
    if st.session_state.log:
        spam_text = " ".join(preprocess_text(msg) for msg, label, _ in st.session_state.log if label == "Spam")
        ham_text = " ".join(preprocess_text(msg) for msg, label, _ in st.session_state.log if label == "Ham")

        if spam_text:
            st.subheader("Spam WordCloud")
            spam_wc = WordCloud(background_color="white", colormap="Reds").generate(spam_text)
            fig3, ax3 = plt.subplots()
            ax3.imshow(spam_wc, interpolation="bilinear")
            ax3.axis("off")
            st.pyplot(fig3)
        else:
            st.info("No spam messages yet.")

        if ham_text:
            st.subheader("Ham WordCloud")
            ham_wc = WordCloud(background_color="white", colormap="Greens").generate(ham_text)
            fig4, ax4 = plt.subplots()
            ax4.imshow(ham_wc, interpolation="bilinear")
            ax4.axis("off")
            st.pyplot(fig4)
        else:
            st.info("No ham messages yet.")
    else:
        st.info("Make predictions to see word clouds.")
