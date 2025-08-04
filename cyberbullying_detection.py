
import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

# Sample cyberbullying dataset (small mockup)
data = {
    'text': [
        "You're such a loser!",
        "Let's meet after school to discuss the project.",
        "Nobody likes you.",
        "You did an amazing job!",
        "You're ugly and stupid.",
        "Have a great weekend ahead.",
        "Go kill yourself.",
        "Good morning, how are you today?"
    ],
    'label': [
        'cyberbullying',
        'non-cyberbullying',
        'cyberbullying',
        'non-cyberbullying',
        'cyberbullying',
        'non-cyberbullying',
        'cyberbullying',
        'non-cyberbullying'
    ]
}

df = pd.DataFrame(data)

# Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    return text

df['clean_text'] = df['text'].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label_encoded'], test_size=0.3, random_state=42
)

# Build pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('clf', LogisticRegression())
])

# Train model
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Basic prevention alert
def check_message(message):
    clean = preprocess_text(message)
    prediction = pipeline.predict([clean])[0]
    label = le.inverse_transform([prediction])[0]
    print(f"Prediction: {label}")
    if label == 'cyberbullying':
        print("⚠️ Alert: Potential cyberbullying detected. Action recommended.")
    else:
        print("✅ Message appears safe.")

# Example usage
check_message("You are pathetic and no one likes you.")
check_message("Great job on your recent assignment!")
