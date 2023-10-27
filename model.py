
import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask, request, jsonify,render_template

def recommend_destinations(user_budget, user_month, user_input):
    import warnings
    warnings.filterwarnings("ignore")

    # Import NLTK packages
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Read your data using pd.read_csv
    df = pd.read_csv("scraped_data (1).csv")

    # Data preprocessing
    df['Known For'] = df['Known For'].str.replace('Known For :', '')
    df['Best Time'] = df['Best Time'].str.replace('Best Time:', '')
    df['About'] = df['About'].str.lower()

    def remove_numbers(text):
        return text.split('.', 1)[1].strip() if '.' in text else text

    df['Title'] = df['Title'].apply(remove_numbers)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
        lemmatized_words = [word for word in lemmatized_words if word not in string.punctuation]
        unique_lemmatized_words = list(set(lemmatized_words))
        return ' '.join(unique_lemmatized_words)

    df['preprocessed_About'] = df['About'].apply(preprocess_text)
    df['keywords'] = df['Known For'] + df['preprocessed_About']
    columns_to_delete = ['Known For', 'About', 'preprocessed_About']
    df = df.drop(columns=columns_to_delete)

    user_budget = float(user_budget)

    def Budget(row):
        if user_budget >= row['Budget']:
            return 1
        else:
            return 0

    df['Budget'] = df.apply(Budget, axis=1)

    user_month = user_month.capitalize()  # Ensure capitalization

    def is_valid_month(month, time_range):
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

        if 'Throughout the year' in time_range:
            return 1
        else:
            words = time_range.split()
            words.remove("to")
            start_month, end_month = [r.strip() for r in words]  # Remove leading/trailing spaces
            start_idx = months.index(start_month)
            end_idx = months.index(end_month)
            user_idx = months.index(month)

            if (start_idx <= end_idx):
                if (start_idx <= user_idx <= end_idx):
                    return 1
                else:
                    return 0
            else:
                if (start_idx >= user_idx >= end_idx):
                    return 0
                else:
                    return 1

    df['ValidMonth'] = df.apply(lambda row: is_valid_month(user_month, row['Best Time']), axis=1)

    new_df = df[(df['Budget'] != 0) & (df['ValidMonth'] != 0)]
    new_df.reset_index(drop=True, inplace=True)

    user_words = word_tokenize(user_input)
    filtered_user_words = [word for word in user_words if word.lower() not in stop_words]

    lemmatized_user_words = [lemmatizer.lemmatize(word) for word in filtered_user_words]
    lemmatized_user_words = [word for word in lemmatized_user_words if word not in string.punctuation]
    unique_lemmatized_user_words = ' '.join(list(set(lemmatized_user_words)))

    vectorizer = CountVectorizer()
    user_keyword_vector = vectorizer.fit_transform([unique_lemmatized_user_words])

    cosine_similarity_scores = []

    for row_index in range(len(new_df)):
        keyword_vector = vectorizer.transform([new_df['keywords'][row_index]])
        cosine_similarity_score = cosine_similarity(user_keyword_vector, keyword_vector)
        cosine_similarity_scores.append(cosine_similarity_score[0][0])

    cosine_similarity_df = pd.DataFrame({
        'Cosine Similarity': cosine_similarity_scores,
        'Keywords': new_df['keywords'],
        'title': new_df['Title']
    })

    cosine_similarity_df = cosine_similarity_df.sort_values(by='Cosine Similarity', ascending=False)

    top_3_similarities = cosine_similarity_df.head(3)

    recommendations = []
    for index, row in top_3_similarities.iterrows():
        recommendations.append(row['title'])

    return recommendations

pickle.dump(recommend_destinations,open("model.pkl","wb"))

app = Flask(__name__,template_folder='templates')
print("hello")
# Load your trained model
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_budget = request.form.get('user_budget')
        user_month = request.form.get('user_month')
        user_input = request.form.get('user_input')

        recommendations = recommend_destinations(user_budget, user_month, user_input)

        return render_template('index.html', recommendations=recommendations)

    except Exception as e:
        return render_template('index.html', error=str(e))


app.run(debug=True)
