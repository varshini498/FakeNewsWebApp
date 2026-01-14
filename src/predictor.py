import pickle
from .preprocessing import clean_text

# Load model and vectorizer
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
model = pickle.load(open("models/model.pkl", "rb"))

def predict_news(text: str):
    """
    Returns:
    - label (0=Fake, 1=Real)
    - confidence score
    - top 5 influential keywords
    """
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    label = model.predict(vector)[0]
    confidence = model.predict_proba(vector)[0].max()

    # Top 5 influential keywords
    feature_names = vectorizer.get_feature_names_out()
    coef = model.coef_[0]
    word_scores = list(zip(feature_names, coef))
    top_keywords = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)[:5]

    return label, confidence, top_keywords
