import re
import numpy as np

# Patterns from global constants
SUSPICIOUS_PATTERNS = [        # Patterns indicating likely fake posts
    r'\\b(work from home|wfh|remote work|online job)\\b',
    r'\\b(immediate (joining|start)|urgently needed)\\b',
    r'\\b(no experience needed|no qualifications)\\b',
    r'\\b(earn (big|money|fast cash)|make money)\\b',
    r'\\b(registration fee|security deposit|investment)\\b',
    r'\\b(government approved|certified|guaranteed)\\b',
    r'\\b(jupiter|saturn|venus|galaxy|universe|cosmic)\\b',
    r'\\b(cash|money|quick rich|fast money)\\b'
]

GIBBERISH_PATTERNS = [         # Common nonsense terms to reject immediately
    r'\\bhaha+\\b',
    r'\\bhihi+\\b',
    r'\\bwow+\\b',
    r'\\basdf+\\b',
    r'\\blorem\\b',
    r'\\btest\\b'
]

def clean_text(text):
    """Clean and normalize text"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Remove URLs/emails
    text = re.sub(r"http\\S+|www\\S+|\\S+@\\S+\\.\\S+", " ", text)
    # Remove special chars except basic punctuation for better feature extraction
    text = re.sub(r"[^a-zA-Z0-9\\s\\.\\,\\!\\?]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def extract_advanced_features(texts):
    """Extract features indicating potential fraud"""
    features = []
    if isinstance(texts, str):
        texts = [texts]
    for text in texts:
        clean_txt = clean_text(text)
        
        # Basic features
        char_len = len(clean_txt)
        word_count = len(clean_txt.split())
        upper_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        
        # New features: special chars and repeated chars
        special_char_count = len(re.findall(r'[^a-zA-Z0-9\\s]', clean_txt))
        repeated_char_count = sum(clean_txt.count(c) > 3 for c in set(clean_txt))
        
        # Ratio of special chars to all chars
        special_char_ratio = special_char_count / max(1, char_len)
        
        # Ratio of digits to all chars
        digit_count = len(re.findall(r'\\d', clean_txt))
        digit_ratio = digit_count / max(1, char_len)
        
        # Content quality features
        stopwords = len([w for w in clean_txt.split() if w in {
            'the', 'and', 'of', 'to', 'in', 'a', 'for', 'with', 'on', 'at'
        }])
        
        # Fraud indicators
        urgency = int(bool(re.search(r'\\b(urgent|immediate|hiring now)\\b', clean_txt)))
        payment = int(bool(re.search(r'\\b(fee|payment|investment|deposit|registration|security deposit)\\b', clean_txt)))
        money = int(bool(re.search(r'\\b(earn|make money|income|profit)\\b', clean_txt)))
        generic_co = int(bool(re.search(r'\\b(jupiter|saturn|galaxy|universe|cosmic)\\b', clean_txt)))
        
        # Count suspicious patterns
        pattern_count = sum(1 for pat in SUSPICIOUS_PATTERNS 
                            if re.search(pat, clean_txt, re.I))
        
        features.append([
            char_len,
            word_count,
            upper_ratio,
            special_char_count,
            repeated_char_count,
            special_char_ratio,
            digit_ratio,
            stopwords,
            urgency,
            payment,
            money,
            generic_co,
            pattern_count
        ])
    return np.array(features)

