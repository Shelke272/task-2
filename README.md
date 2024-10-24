<h3>Input:</h3>

import nltk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

data = {
    'text': [
        "I loved this movie, it was amazing!",
        "The film was boring and too long.",
        "Great acting and an excellent plot, I enjoyed it.",
        "Terrible film, I will never watch it again.",
        "The movie was fantastic, I highly recommend it!",
        "The plot was dull and the characters were uninteresting."
    ],
    'label': [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
}

df = pd.DataFrame(data)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]  # Remove stopwords and non-alphanumeric
    return " ".join(filtered_words)

df['processed_text'] = df['text'].apply(preprocess_text)

tfidf = TfidfVectorizer(max_features=500)  # Limit to 500 features
X = tfidf.fit_transform(df['processed_text']).toarray()
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)


<h3>Output:</h3>

              precision    recall  f1-score   support

         0       1.00      1.00      1.00         1
         1       1.00      1.00      1.00         1

  accuracy                           1.00         2
 macro avg       1.00      1.00      1.00         2

