import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK datasets
nltk.download('stopwords')
nltk.download('punkt')

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    """
    Clean input text by:
    - converting to lowercase
    - removing non-alphabet characters
    - removing stopwords
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return " ".join(tokens)
