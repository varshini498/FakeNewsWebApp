import pickle
from preprocessing import clean_text

# Load saved model and vectorizer
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
model = pickle.load(open("models/model.pkl", "rb"))

def predict_news(text: str):
    """
    Predict if news is REAL or FAKE.
    Returns:
    - label (0=Fake, 1=Real)
    - confidence score
    - top 5 influential keywords
    """
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector)[0].max()
    
    # Get influential keywords
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    word_scores = list(zip(feature_names, coef))
    influential_words = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)[:5]
    
    return prediction, probability, influential_words
