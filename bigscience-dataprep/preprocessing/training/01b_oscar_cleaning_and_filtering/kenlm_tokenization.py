import nltk
import unicodedata
import re
from nltk.tokenize import sent_tokenize

# Ensure you've downloaded the NLTK Punkt tokenizer for sentence splitting.
nltk.download("punkt")


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def normalize_numbers(text):
    return "".join("0" if c.isdigit() else c for c in text)


def normalize_punctuation(text):
    # Adjust these rules as necessary for Azerbaijani text.
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"([.,;:!?])([^\s])", r"\1 \2", text)
    return text


def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = remove_accents(text)  # Remove accents
    text = normalize_numbers(text)  # Normalize numbers
    text = normalize_punctuation(text)  # Normalize punctuation
    return text
