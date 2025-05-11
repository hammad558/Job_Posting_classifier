import pandas as pd
import numpy as np
import os
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import (classification_report, 
                             confusion_matrix, 
                             accuracy_score,
                             roc_auc_score)
from datetime import datetime
from utils import clean_text, extract_advanced_features, SUSPICIOUS_PATTERNS, transform_numeric_features

# ==============================================
# GLOBAL CONSTANTS
# ==============================================
MIN_VALID_LENGTH = 30          # Raised minimum characters for valid job posting
MIN_VALID_WORDS = 6            # Minimum words for valid job posting

GIBBERISH_PATTERNS = [
    r'\bhaha+\b',
    r'\bhihi+\b',
    r'\bwow+\b',
    r'\basdf+\b',
    r'\blorem\b',
    r'\btest\b'
]

# ==============================================
# DATA PROCESSING FUNCTIONS
# ==============================================
def load_dataset(filepath="fake job posting.xlsx"):
    """Load and validate the dataset"""
    print("🔍 Loading dataset...")
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found at: {filepath}\nCurrent working directory: {os.getcwd()}")
        
        print(f"Loading dataset from: {filepath}")
        df = pd.read_excel(filepath)
        print(f"Dataset loaded with shape: {df.shape}")
        
        # Verify required columns
        required_cols = {'title', 'company_profile', 'description', 
                        'requirements', 'benefits', 'fraudulent'}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}\nAvailable columns: {list(df.columns)}")
        
        # Combine text columns
        print("Combining text columns...")
        df['full_text'] = (df['title'].fillna('') + " " +
                          df['company_profile'].fillna('') + " " +
                          df['description'].fillna('') + " " +
                          df['requirements'].fillna('') + " " +
                          df['benefits'].fillna(''))
        
        return df[['full_text', 'fraudulent']].rename(columns={'fraudulent': 'label'})
    
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise

# ==============================================
# MODEL PIPELINE
# ==============================================
def build_enhanced_pipeline():
    """Build the complete fraud detection pipeline with optimized TF-IDF"""
    print("🔧 Building improved model pipeline...")
    
    try:
        # Text features pipeline with enhanced TF-IDF
        text_pipe = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,  # Reduced to avoid overfitting
                stop_words='english',  # Explicit stop words removal
                ngram_range=(1, 2),  # Include bigrams for context
                min_df=5,  # Ignore terms that appear in fewer than 5 documents
                preprocessor=clean_text  # Use custom preprocessing
            ))
        ])
        
        # Numeric features pipeline
        numeric_pipe = Pipeline([
            ('features', FunctionTransformer(transform_numeric_features))
        ])
        
        # Combined pipeline
        pipeline = Pipeline([
            ('features', FeatureUnion([
                ('text', text_pipe),
                ('numeric', numeric_pipe)
            ])),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        return pipeline
    
    except Exception as e:
        print(f"❌ Error building pipeline: {str(e)}")
        raise

# ==============================================
# ENHANCED PREDICTOR CLASS
# ==============================================
class JobPostingDetector:
    def __init__(self, model):
        self.model = model
    
    def predict(self, text):
        """Enhanced prediction with strict rule-based filters"""
        try:
            cleaned = clean_text(text)
            
            # Rule 1: Reject very short text immediately
            if len(cleaned) < MIN_VALID_LENGTH:
                return 1  # Fake
            
            # Rule 2: Reject postings with fewer than minimum words
            if len(cleaned.split()) < MIN_VALID_WORDS:
                return 1  # Fake
            
            # Rule 3: Reject blatant gibberish or nonsense words
            if any(re.search(pat, cleaned, re.I) for pat in GIBBERISH_PATTERNS):
                return 1
            
            # Rule 4: Check suspicious keyword patterns (enhanced)
            if any(re.search(pat, cleaned, re.I) for pat in SUSPICIOUS_PATTERNS):
                return 1
            
            # Rule 5: Use model prediction
            proba = self.model.predict_proba([cleaned])[0][1]
            
            # Lower conservative threshold to improve detection sensitivity
            threshold = 0.22
            
            return 1 if proba > threshold else 0
        
        except Exception as e:
            print(f"❌ Error in prediction: {str(e)}")
            raise

# ==============================================
# EVALUATION & TESTING
# ==============================================
def evaluate_model(model, X, y):
    """Evaluate model performance"""
    print("📊 Evaluating improved model...")
    try:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
        print(f"AUC-ROC: {roc_auc_score(y, y_proba):.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix_improved.png")
        plt.close()
    
    except Exception as e:
        print(f"❌ Error evaluating model: {str(e)}")
        raise

def run_test_cases(detector):
    """Run comprehensive test cases"""
    test_cases = [
        ("wow", 1),
        ("hihi", 1),
        ("asdf asdf asdf", 1),
        ("bank manager at jupiter", 1),
        ("", 1),
        ("senior software engineer at google", 0),
        ("earn $5000 weekly from home", 1),
        ("marketing director at microsoft", 0),
        ("data entry clerk needed immediately", 1),
        ("full stack developer with python experience", 0),
        ("pay $99 to start working today", 1),
        ("registered nurse at cleveland clinic", 0),
        ("Exciting opportunity for a Senior Developer with competitive salary", 0),
        ("No experience needed, earn big money fast", 1)
    ]
    
    print("\n🧪 Running test cases on improved model:")
    print("-"*70)
    all_passed = True
    for text, expected in test_cases:
        try:
            pred = detector.predict(text)
            passed = pred == expected
            all_passed = all_passed and passed
            result = "✅" if passed else "❌"
            print(f"{result} Text: {text.ljust(50)} → Prediction: {'Fake' if pred else 'Real'}, Expected: {'Fake' if expected else 'Real'}")
        except Exception as e:
            print(f"❌ Error in test case '{text}': {str(e)}")
    print("-"*70)
    if all_passed:
        print("All test cases passed.")
    else:
        print("Some test cases failed; consider further tuning.")

# ==============================================
# MAIN TRAINING SCRIPT
# ==============================================
def main():
    try:
        print("\n" + "="*60)
        print("IMPROVED FAKE JOB POSTING DETECTION SYSTEM")
        print("="*60 + "\n")
        
        # Step 1: Load and prepare data
        print("Starting data loading...")
        df = load_dataset()
        print("Cleaning text data...")
        df['cleaned_text'] = df['full_text'].apply(clean_text)
        print(f"Data prepared with shape: {df.shape}")
        
        # Step 2: Split data with stratification to ensure class balance
        print("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
        )
        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Step 3: Build and train improved model
        print("Building pipeline...")
        model = build_enhanced_pipeline()
        print("\n🚀 Training improved model...")
        model.fit(X_train, y_train)
        print("Model training completed.")
        
        # Step 4: Create enhanced detector
        print("Creating detector...")
        detector = JobPostingDetector(model)
        
        # Step 5: Evaluate improved model
        print("Evaluating model on test set...")
        evaluate_model(model, X_test, y_test)
        
        # Step 6: Save model with a timestamped filename for uniqueness
        print("Saving model...")
        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"improved_job_fraud_detector_{timestamp}.pkl"
        model_path = os.path.join("models", model_filename)
        joblib.dump(model, model_path)
        print(f"\n💾 Saved improved model to: {model_path}")
        
        # Step 7: Run test cases on improved model
        print("Running test cases...")
        run_test_cases(detector)
        
        print("\n✅ Improved training and testing completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nTroubleshooting:")
        print(f"1. Verify 'fake job posting.xlsx' exists in: {os.getcwd()}")
        print("2. Check Excel file columns: should include 'title', 'company_profile', 'description', 'requirements', 'benefits', 'fraudulent'")
        print("3. Ensure dependencies installed: pandas, numpy, scikit-learn, joblib, matplotlib, seaborn, openpyxl")
        print("4. Check file permissions for saving the model")
        print("5. Verify available memory and disk space")

if __name__ == "__main__":
    main()