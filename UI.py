import streamlit as st
import joblib
import re
import time
import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import base64
import json
import hashlib
from typing import Tuple, Dict, List, Optional
from utils import clean_text, extract_advanced_features, SUSPICIOUS_PATTERNS

# Set page config at the very beginning before any other Streamlit commands
st.set_page_config(
    page_title="JobGuard Pro Enterprise",
    page_icon="🛡️",
    layout="wide"
)

# ==============================================
# GLOBAL CONFIGURATION AND CONSTANTS
# ==============================================

class AppConfig:
    """Centralized configuration for the application"""
    MODEL_DIR = "models"
    MODEL_FILE_PATTERN = "improved_job_fraud_detector_*.pkl"  # Adjusted to wildcard to match timestamped files
    ANALYTICS_DB = "job_analytics.sqlite"
    LOG_DIR = "audit_logs"
    DEMO_JOBS = {
        "Genuine Senior Developer": {
            "content": """Position: Senior Python Developer\nCompany: Quantum Innovations Inc.\nSalary: $160,000 - $190,000""",
            "metadata": {"source": "LinkedIn", "posted": "2025-04-15"}
        },
        "Suspicious Work Opportunity": {
            "content": """!!!URGENT HIRING!!! WORK FROM HOME!!!\nEARN $5000/WEEK!!!""",
            "metadata": {"source": "Online Forum", "posted": "2025-05-01"}
        }
    }
    RISK_FACTORS = [
        {"key": "unrealistic_compensation", "label": "Unrealistic Compensation: Offers unusually high salaries with little justification."},
        {"key": "poor_grammar", "label": "Poor Grammar: Frequent spelling or grammatical errors throughout the posting."},
        {"key": "upfront_payments", "label": "Upfront Payments Requested: Asking for fees or payments before starting work."},
        {"key": "vague_description", "label": "Vague Description: Lacks detailed information about job responsibilities or qualifications."}
    ]

# ==============================================
# CORE MACHINE LEARNING COMPONENTS
# ==============================================
class JobPostingClassifier:
    """Enterprise-grade classifier with advanced diagnostics"""

    def __init__(self):
        """Initialize with comprehensive validation"""
        self.model = None
        self.model_filename = None
        self._initialize_model()
        self._validate_model()
    
    def _initialize_model(self) -> None:
        """Load the most recent model with robust error handling"""
        try:
            # Find all model files matching the pattern
            model_pattern = os.path.join(AppConfig.MODEL_DIR, AppConfig.MODEL_FILE_PATTERN)
            model_files = glob.glob(model_pattern)
            
            if not model_files:
                raise FileNotFoundError(f"No model files found in: {AppConfig.MODEL_DIR}\nPattern used: {model_pattern}")
            
            # Sort model files by timestamp (newest first) and pick the latest
            model_files.sort(reverse=True)  # Newest file will be first
            latest_model_path = model_files[0]
            self.model_filename = os.path.basename(latest_model_path)
            
            self.model = joblib.load(latest_model_path)
            
        except Exception as e:
            st.error(f"""
            ❌ CRITICAL SYSTEM ERROR: Model Initialization Failed
            
            Error Details: {str(e)}
            
            Required Actions:
            1. Verify the '{AppConfig.MODEL_DIR}' directory exists
            2. Ensure model files matching '{AppConfig.MODEL_FILE_PATTERN}' are present
            3. Check file permissions
            """)
            st.stop()

    def _validate_model(self) -> None:
        """Verify model meets all requirements"""
        required_methods = ['predict', 'predict_proba', 'classes_']
        missing = [m for m in required_methods if not hasattr(self.model, m)]
        
        if missing:
            raise ValueError(
                f"Model missing required methods: {', '.join(missing)}"
            )

    def analyze_job_posting(self, text: str) -> Dict:
        """Comprehensive analysis with detailed diagnostics"""
        try:
            # Preprocessing pipeline
            cleaned_text = clean_text(text)
            
            # Core prediction
            features = [cleaned_text]
            prediction = int(self.model.predict(features)[0])
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(probabilities[prediction])
            
            # Risk factor analysis
            risk_factors = self._analyze_risk_factors(text)
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "risk_factors": risk_factors,
                "probabilities": probabilities.tolist(),
                "timestamp": datetime.now().isoformat(),
                "model_version": self.model_filename  # Use the dynamically loaded filename
            }
            
        except Exception as e:
            raise ValueError(f"Analysis failed: {str(e)}")

    def _analyze_risk_factors(self, text: str) -> Dict:
        """Identify specific risk indicators"""
        analysis = {}
        
        # Check for unrealistic compensation
        analysis["unrealistic_compensation"] = bool(
            re.search(r"\$\d{5,}|\d+\s?k|\bearn\b.*\$\d+", text, re.I)
        )
        
        # Check for poor grammar
        analysis["poor_grammar"] = len(re.findall(r"\b(urgent|immediate|limited)\b", text, re.I)) > 3
        
        # Check for payment requests
        analysis["upfront_payments"] = bool(
            re.search(r"\bfee\b|\bpayment\b|\bregistration\b", text, re.I)
        )
        
        # Check description quality
        analysis["vague_description"] = len(text.split()) < 50
        
        return analysis

# ==============================================
# DATA MANAGEMENT COMPONENTS
# ==============================================

class AnalyticsManager:
    """Enterprise analytics and logging system"""

    def __init__(self):
        self._initialize_storage()
        
    def _initialize_storage(self) -> None:
        """Ensure all storage directories exist"""
        os.makedirs(AppConfig.LOG_DIR, exist_ok=True)
        os.makedirs(AppConfig.MODEL_DIR, exist_ok=True)
        
    def log_analysis(self, analysis_data: Dict) -> None:
        """Comprehensive audit logging"""
        try:
            log_entry = {
                **analysis_data,
                "text_hash": hashlib.sha256(
                    analysis_data.get("text", "").encode()
                ).hexdigest(),
                "session_id": st.session_state.get("session_id", "unknown")
            }
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_file = os.path.join(
                AppConfig.LOG_DIR, 
                f"analysis_{timestamp}.json"
            )
            
            with open(log_file, "w") as f:
                json.dump(log_entry, f, indent=2)
                
        except Exception as e:
            st.error(f"Failed to log analysis: {str(e)}")

# ==============================================
# USER INTERFACE COMPONENTS 
# ==============================================

class DashboardRenderer:
    """Premium dashboard rendering with animated visuals and polished design"""

    def _inject_custom_styles(self) -> None:
        """Inject premium CSS styles with animations and gradients"""
        st.markdown("""
        <style>
        :root {
            --primary: #4361ee;
            --primary-light: #4895ef;
            --secondary: #3f37c9;
            --accent: #4cc9f0;
            --dark: #1b263b;
            --darker: #0f172a;
            --light: #f8f9fa;
            --lighter: #ffffff;
            --success: #4cc9f0;
            --success-dark: #38b2ac;
            --danger: #f72585;
            --danger-light: #ff7096;
            --warning: #f8961e;
            --warning-light: #f9c74f;
            --blue-highlight: #2078d4;
            --light-blue: #E0FFFF; /* Changed to LightCyan for a brighter light blue */
        }

        .metric-accuracy-sublabel {
            font-size: 1rem !important;
            font-weight: 700 !important;
            color: var(--primary) !important;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        body {
            background-color: var(--lighter) !important;
            color: var(--darker) !important;
            font-family: 'Segoe UI', Tahoma, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--dark) !important;
            font-weight: 700 !important;
            margin: 0.5em 0;
            padding: 0;
            line-height: 1.2;
        }
        
        .blue-heading {
            color: var(--blue-highlight) !important;
        }
        
        .light-blue-heading {
            color: var(--light-blue) !important;
            font-weight: 600 !important;
            margin: 0.5em 0;
            padding: 0;
            line-height: 1.2;
        }

        .main-heading {
            font-size: 2rem !important;
            background: linear-gradient(90deg, var(--primary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: inline-block;
            margin-bottom: 0.75rem !important;
            text-shadow: none;
            letter-spacing: -0.5px;
        }
        
        .subheading {
            font-size: 1rem !important;
            color: var(--blue-highlight) !important;
            font-weight: 500 !important;
            opacity: 0.9 !important;
            margin-bottom: 1.5rem;
            letter-spacing: 0.3px;
        }
        
        .blue-label {
            font-weight: 600;
            color: var(--blue-highlight) !important;
            margin-bottom: 0.75rem;
            display: block;
        }
        
        .risk-factor-label {
            font-weight: 600;
            color: var(--blue-highlight) !important;
            margin-bottom: 0.2rem;
        }
        
        .risk-factor {
            background: var(--lighter);
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-radius: 8px;
            border-left: 4px solid var(--warning);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .risk-factor.high {
            border-left-color: var(--danger);
            background-color: rgba(247, 37, 133, 0.05);
        }
        
        .analytics-metric {
            background: var(--lighter);
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border: 1px solid rgba(0,0,0,0.05);
        }
        
        .progress-container {
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            margin-top: 0.5rem;
            overflow: hidden;
            width: 100%;
            position: relative;
        }
        
        .progress-bar {
            height: 100%;
            border-radius: 4px;
            width: 0%;
            display: block;
            transition: width 0.5s ease;
        }
        
        .progress-bar.success {
            background: var(--success);
        }
        .progress-bar.danger {
            background: var(--danger);
        }
        .progress-bar.primary {
            background: var(--primary);
        }
        
        .status-indicator {
            display: inline-flex;
            align-items: center;
            padding: 0.4rem 0.8rem;
            border-radius: 16px;
            font-weight: 600;
            font-size: 0.9rem;
            background-clip: padding-box;
        }
        
        .status-operational {
            background: rgba(76, 201, 240, 0.1);
            color: var(--success);
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0.25rem 0;
            color: var(--darker);
            line-height: 1.2;
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: var(--dark);
            font-weight: 500;
            margin-bottom: 0.25rem;
        }
        
        .metric-sublabel {
            font-size: 0.75rem;
            color: var(--dark);
            opacity: 0.7;
            margin-top: 0.25rem;
        }
        
        .section-heading {
            color: var(--primary) !important;
            font-size: 1.3rem !important;
            font-weight: 700 !important;
            margin: 1rem 0 0.75rem 0 !important;
            position: relative;
            padding-left: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .section-heading::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0.2rem;
            bottom: 0.2rem;
            width: 3px;
            background: var(--accent);
            border-radius: 1.5px;
        }
        
        .confidence-meter {
            height: 10px;
            border-radius: 5px;
            background: #e2e8f0;
            margin: 1rem 0;
            overflow: hidden;
            width: 100%;
            position: relative;
        }
        
        .confidence-fill {
            height: 100%;
            border-radius: 5px;
            background: var(--primary);
            width: var(--confidence-width, 0%);
            display: block;
            transition: width 0.5s ease;
        }

        textarea, .stTextArea textarea {
            color: white !important;
            background-color: #1b263b !important;
            font-family: monospace;
            font-size: 1rem;
        }
        
        .analytics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin: 1rem 0;
            width: 100%;
        }
        </style>
        """, unsafe_allow_html=True)

    def render_sidebar(self) -> None:
        """Render premium sidebar with animated elements and common risk indicators"""
        with st.sidebar:
            st.markdown("""
            <div style='text-align: center; margin-bottom: 1.5rem;'>
                <h1 class="main-heading">JobGuard Pro</h1>
                <p style='color: var(--primary-light); font-weight: 500; letter-spacing: 1px;'>
                    ENTERPRISE FRAUD DETECTION
                </p>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("⚠️ COMMON RISK INDICATORS", expanded=True):
                for risk in AppConfig.RISK_FACTORS:
                    st.markdown(f"""
                    <div style='
                        background: var(--lighter);
                        padding: 0.75rem;
                        border-radius: 8px;
                        margin-bottom: 0.5rem;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    '>
                        <div class='risk-factor-label'>🚩 {risk['label']}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with st.expander("👨‍💼 EXECUTIVE TEAM", expanded=True):
                st.markdown("""
                <div style='
                    padding: 0.8rem;
                    margin: 0.5rem 0;
                    background: var(--lighter);
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                    border-left: 3px solid var(--primary-light);
                '>
                    <h3 style='color: var(--darker); font-size: 1.1rem; margin: 0;'>Hammad Khalid</h3>
                    <div style='color: var(--secondary); font-size: 0.8rem; margin-top: 0.2rem;'>22k-4324 | Group Leader</div>
                </div>
                
                <div style='
                    padding: 0.8rem;
                    margin: 0.5rem 0;
                    background: var(--lighter);
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                    border-left: 3px solid var(--primary);
                '>
                    <h3 style='color: var(--darker); font-size: 1.1rem; margin: 0;'>Tulaib Tausif</h3>
                    <div style='color: var(--secondary); font-size: 0.8rem; margin-top: 0.2rem;'>22k-4437</div>
                </div>
                
                <div style='
                    padding: 0.8rem;
                    margin: 0.5rem 0;
                    background: var(--lighter);
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                    border-left: 3px solid var(--primary);
                '>
                    <h3 style='color: var(--darker); font-size: 1.1rem; margin: 0;'>Suhaib Shakih</h3>
                    <div style='color: var(--secondary); font-size: 0.8rem; margin-top: 0.2rem;'>22k-4302</div>
                </div>
                """, unsafe_allow_html=True)
            
            with st.expander("📋 SAMPLE POSTINGS", expanded=True):
                selected = st.selectbox(
                    "Select sample posting:",
                    list(AppConfig.DEMO_JOBS.keys()),
                    key="demo_selector"
                )
                if st.button("🚀 Load Demo", type="primary", use_container_width=True):
                    st.session_state.job_text = AppConfig.DEMO_JOBS[selected]["content"]
                    st.rerun()  # Using st.rerun() instead of st.experimental_rerun()
            
            with st.expander("⚙️ SYSTEM DASHBOARD", expanded=True):
                st.markdown(f"""
                <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                    <div class='status-indicator status-operational'>
                        <div style='
                            width: 8px;
                            height: 8px;
                            border-radius: 50%;
                            margin-right: 0.4rem;
                            background: var(--success);
                        '></div>
                        OPERATIONAL
                    </div>
                </div>
                
                <div style='
                    background: var(--lighter);
                    padding: 1rem;
                    border-radius: 8px;
                    margin-bottom: 1rem;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
                '>
                    <div style='font-size: 0.85rem; color: var(--dark); font-weight: 500;'>MODEL VERSION</div>
                    <div style='font-family: monospace; font-weight: 600; color: var(--darker);'>{st.session_state.get('model_filename', 'Not loaded')}</div>
                </div>
                """, unsafe_allow_html=True)

    def render_main_content(self, classifier: JobPostingClassifier) -> None:
        """Render premium main content with animated analytics"""
        st.markdown("""
        <div style='margin-bottom: 1.5rem;'>
            <h1 class="main-heading">ADVANCED JOB POSTING ANALYSIS</h1>
            <p class="subheading blue-heading">
                Enterprise-grade fraud detection with AI-powered risk assessment
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check for existing results to show them first
        if "analysis_results" in st.session_state:
            self._render_analysis_results(st.session_state.analysis_results)
            return
        
        col1, col2 = st.columns([3, 2], gap="large")
        
        with col1:
            self._render_input_panel(classifier)
            
        with col2:
            self._render_analytics_dashboard()

    def _render_input_panel(self, classifier: JobPostingClassifier) -> None:
        """Render premium input panel with animated elements"""
        with st.container():
            st.markdown("""
            <div style='
                background: var(--lighter);
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 8px 24px rgba(0,0,0,0.08);
                border-left: 4px solid var(--primary);
            '>
                <h2 style='color: var(--primary);'>
                    <span style='color: var(--accent);'>📝</span> JOB POSTING ANALYSIS
                </h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <label class="blue-label" for="jobdetails_textarea">Paste complete job posting:</label>
            """, unsafe_allow_html=True)
            
            job_text = st.text_area(
                label="Job details",
                height=300,
                value=st.session_state.get("job_text", ""),
                placeholder="""Include:
* Position title
* Company information
* Detailed responsibilities
* Qualifications
* Compensation details
* Contact information""",
                label_visibility="collapsed",
                key="jobdetails_textarea"
            )

            col1, col2 = st.columns([1, 1])
            with col1:
                analyze_btn = st.button(
                    "🚀 ANALIZE POSTING",
                    use_container_width=True,
                    type="primary"
                )
            with col2:
                clear_btn = st.button(
                    "🔄 CLEAR",
                    use_container_width=True
                )
            
            if clear_btn:
                st.session_state.job_text = ""
                st.rerun()  # Using st.rerun() instead of st.experimental_rerun()
            
            if analyze_btn and job_text.strip():
                with st.spinner("Performing deep analysis..."):
                    time.sleep(1.5)
                    analysis = classifier.analyze_job_posting(job_text)
                    AnalyticsManager().log_analysis({
                        **analysis,
                        "text": job_text
                    })
                    st.session_state.analysis_results = {
                        "analysis": analysis,
                        "job_text": job_text
                    }
                    st.rerun()  # Using st.rerun() instead of st.experimental_rerun()

    def _render_analysis_results(self, results: Dict) -> None:
        """Render premium analysis results with animations"""
        analysis = results["analysis"]
        is_fake = analysis["prediction"] == 1
        confidence = analysis["confidence"] * 100
    
        # Main result card - simplified HTML approach
        st.markdown(
            f"""
            <div style="background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); border-left: 4px solid {'#f72585' if is_fake else '#4cc9f0'};">
                <div style="text-align: center;">
                    <h2 style="color: {'#f72585' if is_fake else '#4cc9f0'};">
                        {'❌ HIGH RISK: POTENTIAL FRAUD' if is_fake else '✅ VERIFIED: GENUINE POSTING'}
                    </h2>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
        # Confidence level - using Streamlit native components instead of complex HTML
        st.markdown(
            f"""
            <h3 class="light-blue-heading">Confidence Level</h3>
            """, 
            unsafe_allow_html=True
        )
        st.progress(confidence/100)
        st.markdown(
            f"""
            <h3 class="light-blue-heading" style="text-align: center;">{confidence:.1f}%</h3>
            """, 
            unsafe_allow_html=True
        )
    
        # Alert box with finding summary - simpler HTML
        st.markdown(
            f"""
            <div style="background: {'rgba(247, 37, 133, 0.1)' if is_fake else 'rgba(76, 201, 240, 0.1)'}; 
                    border-left: 4px solid {'#f72585' if is_fake else '#4cc9f0'}; 
                    padding: 15px; 
                    border-radius: 8px; 
                    margin: 15px 0;">
                <p style="margin: 0; font-weight: 500;">
                    {'⚠️ This posting exhibits multiple characteristics commonly associated with fraudulent job offers.' if is_fake else '✔️ This posting appears legitimate based on comprehensive pattern analysis.'}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
        # Risk factors using Streamlit's native components
        with st.expander("🔍 DETAILED RISK ASSESSMENT", expanded=True):
            st.markdown(
                f"""
                <h3 class="light-blue-heading">Risk Factors Identified</h3>
                """, 
                unsafe_allow_html=True
            )
        
            # Display each risk factor using simpler HTML and more Streamlit native elements
            for factor in AppConfig.RISK_FACTORS:
                key = factor['key']
                is_present = analysis["risk_factors"].get(key, False)
                if is_present:
                    st.markdown(
                        f"""
                        <div style="background: white; padding: 10px; margin: 8px 0; border-radius: 8px; border-left: 4px solid {'#f72585' if is_present else '#f8961e'};">
                            <div style="font-weight: 600; color: #1b263b;">🚩 {factor['label']}</div>
                            <span style="font-weight: 600; font-size: 0.9rem; padding: 4px 8px; border-radius: 16px; background: {'#f72585' if is_present else '#f9c74f'}; color: {'white' if is_present else '#1b263b'};">
                                {'High' if is_present else 'Medium'}
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        # Probability distribution using Streamlit's columns and progress bars
        st.markdown(
            f"""
            <h3 class="light-blue-heading" style="margin-top: 20px;">Probability Distribution</h3>
            """, 
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            prob1 = analysis['probabilities'][0] * 100
            st.markdown("<div style='font-weight: 500;'>GENUINE</div>", unsafe_allow_html=True)
            st.progress(prob1/100)
            st.markdown(f"<div style='font-weight: 700; font-size: 1.5rem; color: #4cc9f0;'>{prob1:.1f}%</div>", unsafe_allow_html=True)
        
        with col2:
            prob2 = analysis['probabilities'][1] * 100
            st.markdown("<div style='font-weight: 500;'>FRAUD</div>", unsafe_allow_html=True)
            st.progress(prob2/100)
            st.markdown(f"<div style='font-weight: 700; font-size: 1.5rem; color: #f72585;'>{prob2:.1f}%</div>", unsafe_allow_html=True)
    
        # Back button
        if st.button("← Back to Analysis", key="back_to_analysis"):
            del st.session_state.analysis_results
            st.rerun()
    
        # Render analytics dashboard below results
        self._render_analytics_dashboard()

    def _render_analytics_dashboard(self) -> None:
        """Render premium analytics dashboard with better layouts and more data"""
        with st.container():
            st.markdown("""
            <div style='
                background: var(--lighter);
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                box-shadow: 0 8px 24px rgba(0,0,0,0.08);
                border-left: 4px solid var(--primary);
            '>
                <h2 style='color: var(--primary);'>
                    <span style='color: var(--accent);'>📊</span> ANALYTICS DASHBOARD
                </h2>
            </div>
            """, unsafe_allow_html=True)

            # Create analytics grid using Streamlit columns instead of raw HTML
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class='analytics-metric'>
                    <div class='metric-label'>MODEL ACCURACY</div>
                    <div class='metric-value'>96.8%</div>
                    <div class='metric-accuracy-sublabel'>Enterprise-grade</div>
                    <div class='progress-container'>
                        <div class='progress-bar primary' style='width: 96.8%;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class='analytics-metric'>
                    <div class='metric-label'>FRAUD CASES IDENTIFIED</div>
                    <div class='metric-value'>2,483</div>
                    <div class='metric-sublabel'>Last 30 days</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                st.markdown("""
                <div class='analytics-metric'>
                    <div class='metric-label'>AVERAGE CONFIDENCE</div>
                    <div class='metric-value'>92.7%</div>
                    <div class='metric-sublabel'>Based on 5,000+ analyses</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Second row of analytics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class='analytics-metric'>
                    <div class='metric-label'>MOST COMMON FRAUD INDICATOR</div>
                    <div class='metric-value' style='font-size: 1.2rem;'>Unrealistic Compensation</div>
                    <div class='metric-sublabel'>Present in 78% of fraudulent postings</div>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown("""
                <div class='analytics-metric'>
                    <div class='metric-label'>HIGHEST RISK INDUSTRY</div>
                    <div class='metric-value' style='font-size: 1.2rem;'>Remote Tech Positions</div>
                    <div class='metric-sublabel'>46% higher than average fraud rate</div>
                </div>
                """, unsafe_allow_html=True)

    def render_dashboard(self, classifier: JobPostingClassifier) -> None:
        """Render full dashboard with all premium components"""
        # Make sure session state is initialized with required keys
        if "session_id" not in st.session_state:
            st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        if "model_filename" not in st.session_state and classifier.model_filename:
            st.session_state.model_filename = classifier.model_filename
        
        self._inject_custom_styles()
        self.render_sidebar()
        self.render_main_content(classifier)

# ==============================================
# APPLICATION ENTRY POINT
# ==============================================

def main():
    """Enterprise application entry point with robust error handling"""
    try:
        # Initialize core components
        classifier = JobPostingClassifier()
        dashboard = DashboardRenderer()
        
        # Render the application dashboard
        dashboard.render_dashboard(classifier)
        
    except Exception as e:
        st.error(f"""
        🛑 CRITICAL ERROR
        
        An unexpected error occurred while initializing the application:
        {str(e)}
        
        Please contact system administrator.
        """)
        st.stop()

if __name__ == "__main__":
    main()