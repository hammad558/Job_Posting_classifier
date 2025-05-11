import re
import numpy as np  # Add this import for transform_numeric_features
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text):
    """Clean and preprocess text for analysis"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs and special chars (preserving your original logic)
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z\s]", " ", text)
    
    # Tokenize and remove stop words
    words = text.split()
    stop_words = set(ENGLISH_STOP_WORDS)
    words = [word for word in words if word not in stop_words]
    
    # Basic normalization (remove common suffixes as a substitute for lemmatization/stemming)
    normalized_words = []
    for word in words:
        word = re.sub(r'ing\b', '', word)  # Remove -ing
        word = re.sub(r'ed\b', '', word)   # Remove -ed
        word = re.sub(r'es\b|s\b', '', word)  # Remove -es or -s
        normalized_words.append(word.strip())
    
    # Join words back and collapse whitespace
    text = " ".join(normalized_words)
    text = re.sub(r"\s+", " ", text)  # Replace multiple spaces with single space
    
    return text.strip()

# Shared constants
SUSPICIOUS_PATTERNS = [
    r'\b(work from home|wfh|remote work|online job)\b',
    r'\b(immediate (joining|start)|urgently needed)\b',
    r'\b(no experience needed|no qualifications)\b',
    r'\b(earn (big|money|fast cash)|make money)\b',
    r'\b(registration fee|security deposit|investment)\b',
    r'\b(government approved|certified|guaranteed)\b',
    r'\b(jupiter|saturn|venus|galaxy|universe|cosmic)\b',
    r'\b(cash|money|quick rich|fast money)\b'
]

def extract_advanced_features(text):
    """Extract features indicating potential fraud"""
    clean_txt = clean_text(text)
    
    # Basic features
    char_len = len(clean_txt)
    word_count = len(clean_txt.split())
    upper_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
    
    # New features
    special_char_count = len(re.findall(r'[^a-zA-Z0-9\s]', clean_txt))
    repeated_char_count = sum(clean_txt.count(c) > 3 for c in set(clean_txt))
    special_char_ratio = special_char_count / max(1, char_len)
    digit_count = len(re.findall(r'\d', clean_txt))
    digit_ratio = digit_count / max(1, char_len)
    
    # Content quality
    stopwords = len([w for w in clean_txt.split() if w in ENGLISH_STOP_WORDS])
    
    # Fraud indicators
    urgency = int(bool(re.search(r'\b(urgent|immediate|hiring now)\b', clean_txt)))
    payment = int(bool(re.search(r'\b(fee|payment|investment|deposit|registration)\b', clean_txt)))
    money = int(bool(re.search(r'\b(earn|make money|income|profit)\b', clean_txt)))
    generic_co = int(bool(re.search(r'\b(jupiter|saturn|galaxy|universe)\b', clean_txt)))
    pattern_count = sum(1 for pat in SUSPICIOUS_PATTERNS if re.search(pat, clean_txt, re.I))
    
    return [
        char_len, word_count, upper_ratio,
        special_char_count, repeated_char_count,
        special_char_ratio, digit_ratio,
        stopwords, urgency, payment,
        money, generic_co, pattern_count
    ]

def transform_numeric_features(texts):
    """Transform text data into numeric features"""
    return np.array([extract_advanced_features(text) for text in texts])