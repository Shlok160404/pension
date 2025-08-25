#tools.py
"""
Enhanced Pension AI Tools with Merged Features

This file combines the advanced role-based context detection and regulator tools
from the current version with enhanced features from the uploaded version:

MERGED FEATURES:
1. Enhanced ML-based fraud detection with XGBoost model fallback
2. Better projection service integration with the projection agent
3. Improved context management with cleaner fallback mechanisms
4. Enhanced error handling and logging throughout all tools

ENHANCED TOOLS:
- analyze_risk_profile: Role-based context detection with advisor/regulator support
- detect_fraud: ML model + rule-based fallback with comprehensive feature extraction
- project_pension: Integration with projection service for more accurate calculations
- knowledge_base_search: Dual search (general + user documents) with role-based access
- analyze_uploaded_document: Enhanced document analysis with better error handling
- Regulator tools: System-wide analysis capabilities for oversight

CONTEXT MANAGEMENT:
- Request-scoped context (production)
- Global fallback variables
- Thread-local storage (testing)
- Role-based access control
"""

import json
from typing import Dict, Any, List, Optional

from sqlalchemy.orm import Session
from langchain.tools import tool
from pydantic.v1 import BaseModel, Field, validator, root_validator
import re

from ..database import SessionLocal
from .. import models
from ..chromadb_service import get_or_create_collection, query_collection
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Set Google API key for LangChain with fallback
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    os.environ["GOOGLE_API_KEY"] = gemini_key
else:
    # Set a dummy key for testing (will fail gracefully)
    os.environ["GOOGLE_API_KEY"] = "dummy_key_for_testing"

json_llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    response_mime_type="application/json"
)

# --- Tool 1: Risk Analysis ---
class RiskToolInput(BaseModel):
    user_id: Optional[int] = Field(description="The numeric database ID for the user. If not provided, will be retrieved from current session.")
    query: Optional[str] = Field(description="The user's original query for context and role-based detection.")

    @validator("user_id", pre=True)
    def coerce_user_id(cls, value):
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        return None

@tool(args_schema=RiskToolInput)
def analyze_risk_profile(user_id: int = None) -> Dict[str, Any]:
    """
    Analyzes a user's risk profile based on their ID by fetching their data
    and evaluating it against fixed financial risk factors.
    Returns a structured JSON object with the complete risk assessment.
    """
    print(f"🔍 Tool Debug: analyze_risk_profile called with user_id={user_id}")
    
    # PRIORITY 1: Get current user's ID from request context (most secure)
    current_user_id = get_current_user_id_from_context()
    if not current_user_id:
        return {"error": "User not authenticated. Please log in."}
    
    # PRIORITY 2: Get target user_id from input (could be client ID for advisors)
    target_user_id_input = None
    if user_id is not None:
        if isinstance(user_id, str):
            target_user_id_input = extract_user_id_from_input(user_id)
        else:
            target_user_id_input = user_id
    
    print(f"\n--- TOOL: Running Risk Analysis ---")
    print(f"🔍 Context: Current user ID: {current_user_id}")
    print(f"🔍 Context: Target user ID from input: {target_user_id_input}")
    
    db: Session = SessionLocal()
    try:
        # Apply role-based context detection using workflow context
        original_query = get_current_query_from_context() or ""
        target_user_id, context_type = detect_role_based_context(original_query, current_user_id, db)
        if context_type == 'client':
            print(f"🔍 Role Context: Accessing client data (ID: {target_user_id})")
            final_user_id = target_user_id
        elif context_type == 'self':
            print(f"🔍 Role Context: Accessing own data (ID: {target_user_id})")
            final_user_id = target_user_id
        else:
            print(f"🔍 Role Context: Unknown context, using current user ID: {current_user_id}")
            final_user_id = current_user_id
        
        pension_data = db.query(models.PensionData).filter(models.PensionData.user_id == final_user_id).first()
        if not pension_data:
            return {"error": f"No pension data found for User ID: {final_user_id}"}

        user_data = {
            "Annual_Income": pension_data.annual_income,
            "Debt_Level": pension_data.debt_level,
            "Risk_Tolerance": pension_data.risk_tolerance,
            "Volatility": pension_data.volatility,
            "Portfolio_Diversity_Score": pension_data.portfolio_diversity_score,
            "Health_Status": pension_data.health_status
        }
        prompt = f"""
        **SYSTEM:** You are a Methodical Financial Risk Analyst System...
        **TASK:** Analyze the user's data below...
        **RISK ANALYSIS FACTORS:**
        1.  **Market Risk Mismatch**: `Risk_Tolerance` is 'Low' but `Volatility` > 3.5.
        2.  **Concentration Risk**: `Portfolio_Diversity_Score` < 0.5.
        3.  **High Debt-to-Income Ratio**: `Debt_Level` > 50% of `Annual_Income`.
        4.  **Longevity & Health Risk**: `Health_Status` is 'Poor'.
        **DATA TO ANALYZE:**
        ```json
        {json.dumps(user_data, indent=2)}
        ```
        **OUTPUT INSTRUCTIONS:**
        Return a single JSON object with this structure: {{"risk_level": "Low/Medium/High", "risk_score": float, "positive_factors": [], "risks_identified": [], "summary": "..."}}
        """
        response = json_llm.invoke(prompt)
        result = json.loads(response.content)
        
        # Add data source information
        result["data_source"] = "DATABASE_PENSION_DATA"
        
        # Use role-aware language based on context
        original_query = get_current_query_from_context() or ""
        _, context_type = detect_role_based_context(original_query, current_user_id, db)
        
        if context_type == 'client':
            result["note"] = "This risk analysis is based on the client's pension data stored in our database, not from uploaded documents."
        elif context_type == 'self':
            result["note"] = "This risk analysis is based on your pension data stored in our database, not from uploaded documents."
        else:
            result["note"] = "This risk analysis is based on the user's pension data stored in our database, not from uploaded documents."
        
        return result
    finally:
        db.close()

# --- Tool 2: Fraud Detection ---
class FraudToolInput(BaseModel):
    user_id: Optional[int] = Field(description="The numeric database ID for the user. If not provided, will be retrieved from current session.")
    query: Optional[str] = Field(description="The user's original query for context and role-based detection.")

    @validator("user_id", pre=True)
    def coerce_user_id(cls, value):
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        return None

@tool(args_schema=FraudToolInput)
def detect_fraud(user_id: int = None) -> Dict[str, Any]:
    """
    Analyzes a user's recent transactions based on their ID to detect potential fraud.
    Uses ML model fallback for more accurate detection when available.
    Evaluates data against fixed rules and returns a structured JSON assessment.
    """
    # PRIORITY 1: Get current user's ID from request context (most secure)
    current_user_id = get_current_user_id_from_context()
    if not current_user_id:
        return {"error": "User not authenticated. Please log in."}
    
    # PRIORITY 2: Get target user_id from input (could be client ID for advisors)
    target_user_id_input = None
    if user_id is not None:
        if isinstance(user_id, str):
            target_user_id_input = extract_user_id_from_input(user_id)
        else:
            target_user_id_input = user_id
    
    print(f"\n--- TOOL: Running Fraud Detection ---")
    print(f"🔍 Context: Current user ID: {current_user_id}")
    print(f"🔍 Context: Target user ID from input: {target_user_id_input}")
    
    db: Session = SessionLocal()
    try:
        # Apply role-based context detection using workflow context
        original_query = get_current_query_from_context() or ""
        target_user_id, context_type = detect_role_based_context(original_query, current_user_id, db)
        if context_type == 'client':
            print(f"🔍 Role Context: Accessing client data (ID: {target_user_id})")
            final_user_id = target_user_id
        elif context_type == 'self':
            print(f"🔍 Role Context: Accessing own data (ID: {target_user_id})")
            final_user_id = target_user_id
        else:
            print(f"🔍 Role Context: Unknown context, using current user ID: {current_user_id}")
            final_user_id = current_user_id
        
        pension_data = db.query(models.PensionData).filter(models.PensionData.user_id == final_user_id).first()
        if not pension_data:
            return {"error": f"No pension data found for User ID: {final_user_id}"}
        
        user_data = {
            "Country": pension_data.country,
            "Transaction_Amount": pension_data.transaction_amount,
            "Suspicious_Flag": pension_data.suspicious_flag,
            "Anomaly_Score": pension_data.anomaly_score,
            "Geo_Location": pension_data.geo_location,
            "Age": pension_data.age,
            "Annual_Income": pension_data.annual_income,
            "Current_Savings": pension_data.current_savings,
            "Retirement_Age_Goal": pension_data.retirement_age_goal,
            "Contribution_Amount": pension_data.contribution_amount,
            "Employer_Contribution": pension_data.employer_contribution,
            "Total_Annual_Contribution": pension_data.total_annual_contribution,
            "Years_Contributed": pension_data.years_contributed,
            "Annual_Return_Rate": pension_data.annual_return_rate,
            "Volatility": pension_data.volatility,
            "Fees_Percentage": pension_data.fees_percentage,
            "Projected_Pension_Amount": pension_data.projected_pension_amount,
            "Expected_Annual_Payout": pension_data.expected_annual_payout,
            "Inflation_Adjusted_Payout": pension_data.inflation_adjusted_payout,
            "Years_of_Payout": pension_data.years_of_payout,
            "Number_of_Dependents": pension_data.number_of_dependents,
            "Life_Expectancy_Estimate": pension_data.life_expectancy_estimate,
            "Debt_Level": pension_data.debt_level,
            "Monthly_Expenses": pension_data.monthly_expenses,
            "Savings_Rate": pension_data.savings_rate,
            "Portfolio_Diversity_Score": pension_data.portfolio_diversity_score,
            "Transaction_Pattern_Score": pension_data.transaction_pattern_score,
            "Account_Age": pension_data.account_age,
            "Gender": pension_data.gender,
            "Employment_Status": pension_data.employment_status,
            "Risk_Tolerance": pension_data.risk_tolerance,
            "Contribution_Frequency": pension_data.contribution_frequency,
            "Investment_Type": pension_data.investment_type,
            "Survivor_Benefits": pension_data.survivor_benefits,
            "Marital_Status": pension_data.marital_status,
            "Education_Level": pension_data.education_level,
            "Health_Status": pension_data.health_status,
            "Home_Ownership_Status": pension_data.home_ownership_status,
            "Investment_Experience_Level": pension_data.investment_experience_level,
            "Financial_Goals": pension_data.financial_goals,
            "Insurance_Coverage": pension_data.insurance_coverage,
            "Tax_Benefits_Eligibility": pension_data.tax_benefits_eligibility,
            "Government_Pension_Eligibility": pension_data.government_pension_eligibility,
            "Private_Pension_Eligibility": pension_data.private_pension_eligibility,
            "Pension_Type": pension_data.pension_type,
            "Withdrawal_Strategy": pension_data.withdrawal_strategy,
            "Transaction_Channel": pension_data.transaction_channel,
            "Previous_Fraud_Flag": pension_data.previous_fraud_flag
        }
        
        # TRY ML MODEL FIRST (MORE ACCURATE)
        try:
            import joblib
            import os
            
            # Look for the fraud model in multiple locations
            model_paths = [
                "fraud_final_model.joblib",  # Current directory
                "../fraud_final_model.joblib",  # Parent directory
                "app/fraud_final_model.joblib",  # App directory
                os.path.join(os.path.dirname(__file__), "..", "fraud_final_model.joblib")  # Relative to tools
            ]
            
            fraud_model = None
            for path in model_paths:
                if os.path.exists(path):
                    print(f"🔍 Loading fraud model from: {path}")
                    fraud_model = joblib.load(path)
                    break
            
            if fraud_model:
                print("🚀 Using ML model for fraud detection")
                
                # Extract model components from the dictionary
                if isinstance(fraud_model, dict) and 'model' in fraud_model:
                    xgb_model = fraud_model['model']
                    scaler = fraud_model['scaler']
                    training_columns = fraud_model.get('training_columns', [])
                    print(f"📊 Model components: XGBoost classifier, StandardScaler, {len(training_columns)} training columns")
                else:
                    # Direct model (fallback)
                    xgb_model = fraud_model
                    scaler = None
                    training_columns = []
                
                # Prepare features for ML model
                # Create a 69-feature vector matching the training data structure
                features = [0.0] * 69  # Initialize with zeros
                
                # Numeric features (direct mapping)
                features[0] = float(user_data.get("Age", 0) or 0)  # Age
                features[1] = float(user_data.get("Annual_Income", 0) or 0)  # Annual_Income
                features[2] = float(user_data.get("Current_Savings", 0) or 0)  # Current_Savings
                features[3] = float(user_data.get("Retirement_Age_Goal", 0) or 0)  # Retirement_Age_Goal
                features[4] = float(user_data.get("Contribution_Amount", 0) or 0)  # Contribution_Amount
                features[5] = float(user_data.get("Employer_Contribution", 0) or 0)  # Employer_Contribution
                features[6] = float(user_data.get("Total_Annual_Contribution", 0) or 0)  # Total_Annual_Contribution
                features[7] = float(user_data.get("Years_Contributed", 0) or 0)  # Years_Contributed
                features[8] = float(user_data.get("Annual_Return_Rate", 0) or 0)  # Annual_Return_Rate
                features[9] = float(user_data.get("Volatility", 0) or 0)  # Volatility
                features[10] = float(user_data.get("Fees_Percentage", 0) or 0)  # Fees_Percentage
                features[11] = float(user_data.get("Projected_Pension_Amount", 0) or 0)  # Projected_Pension_Amount
                features[12] = float(user_data.get("Expected_Annual_Payout", 0) or 0)  # Expected_Annual_Payout
                features[13] = float(user_data.get("Inflation_Adjusted_Payout", 0) or 0)  # Inflation_Adjusted_Payout
                features[14] = float(user_data.get("Years_of_Payout", 0) or 0)  # Years_of_Payout
                features[15] = float(user_data.get("Transaction_Amount", 0) or 0)  # Transaction_Amount
                features[16] = float(user_data.get("Anomaly_Score", 0) or 0)  # Anomaly_Score
                features[17] = float(user_data.get("Number_of_Dependents", 0) or 0)  # Number_of_Dependents
                features[18] = float(user_data.get("Life_Expectancy_Estimate", 0) or 0)  # Life_Expectancy_Estimate
                features[19] = float(user_data.get("Debt_Level", 0) or 0)  # Debt_Level
                features[20] = float(user_data.get("Monthly_Expenses", 0) or 0)  # Monthly_Expenses
                features[21] = float(user_data.get("Savings_Rate", 0) or 0)  # Savings_Rate
                features[22] = float(user_data.get("Portfolio_Diversity_Score", 0) or 0)  # Portfolio_Diversity_Score
                features[23] = float(user_data.get("Transaction_Pattern_Score", 0) or 0)  # Transaction_Pattern_Score
                features[24] = float(user_data.get("Account_Age", 0) or 0)  # Account_Age
                
                # Categorical features (one-hot encoding)
                # Gender encoding
                gender = user_data.get("Gender", "").lower()
                if gender == "male":
                    features[25] = 1.0  # Gender_Male
                elif gender == "other":
                    features[26] = 1.0  # Gender_Other
                # Female is default (all 0)
                
                # Country encoding
                country = user_data.get("Country", "").lower()
                if country == "canada":
                    features[27] = 1.0  # Country_Canada
                elif country == "germany":
                    features[28] = 1.0  # Country_Germany
                elif country == "uk":
                    features[29] = 1.0  # Country_UK
                elif country == "usa" or country == "us":
                    features[30] = 1.0  # Country_USA
                # Default country is 0
                
                # Employment status encoding
                emp_status = user_data.get("Employment_Status", "").lower()
                if "part" in emp_status:
                    features[31] = 1.0  # Employment_Status_Part-time
                elif "retire" in emp_status:
                    features[32] = 1.0  # Employment_Status_Retired
                elif "self" in emp_status:
                    features[33] = 1.0  # Employment_Status_Self-employed
                elif "unemploy" in emp_status:
                    features[34] = 1.0  # Employment_Status_Unemployed
                # Full-time is default (all 0)
                
                # Risk tolerance encoding
                risk_tol = user_data.get("Risk_Tolerance", "").lower()
                if risk_tol == "low":
                    features[35] = 1.0  # Risk_Tolerance_Low
                elif risk_tol == "medium":
                    features[36] = 1.0  # Risk_Tolerance_Medium
                # High is default (all 0)
                
                # Contribution frequency encoding
                contrib_freq = user_data.get("Contribution_Frequency", "").lower()
                if contrib_freq == "monthly":
                    features[37] = 1.0  # Contribution_Frequency_Monthly
                elif contrib_freq == "quarterly":
                    features[38] = 1.0  # Contribution_Frequency_Quarterly
                # Annual is default (all 0)
                
                # Investment type encoding
                inv_type = user_data.get("Investment_Type", "").lower()
                if inv_type == "etf":
                    features[39] = 1.0  # Investment_Type_ETF
                elif inv_type == "mutual fund":
                    features[40] = 1.0  # Investment_Type_Mutual Fund
                elif inv_type == "real estate":
                    features[41] = 1.0  # Investment_Type_Real Estate
                elif inv_type == "stocks":
                    features[42] = 1.0  # Investment_Type_Stocks
                # Bonds is default (all 0)
                
                # Survivor benefits encoding
                if user_data.get("Survivor_Benefits", False):
                    features[43] = 1.0  # Survivor_Benefits_Yes
                
                # Marital status encoding
                marital = user_data.get("Marital_Status", "").lower()
                if marital == "married":
                    features[44] = 1.0  # Marital_Status_Married
                elif marital == "single":
                    features[45] = 1.0  # Marital_Status_Single
                elif marital == "widowed":
                    features[46] = 1.0  # Marital_Status_Widowed
                # Divorced is default (all 0)
                
                # Education level encoding
                education = user_data.get("Education_Level", "").lower()
                if "high school" in education:
                    features[47] = 1.0  # Education_Level_High School
                elif "master" in education:
                    features[48] = 1.0  # Education_Level_Master's
                elif "phd" in education:
                    features[49] = 1.0  # Education_Level_PhD
                # Bachelor's is default (all 0)
                
                # Health status encoding
                health = user_data.get("Health_Status", "").lower()
                if health == "good":
                    features[50] = 1.0  # Health_Status_Good
                elif health == "poor":
                    features[51] = 1.0  # Health_Status_Poor
                # Excellent is default (all 0)
                
                # Home ownership encoding
                home_own = user_data.get("Home_Ownership_Status", "").lower()
                if home_own == "own":
                    features[52] = 1.0  # Home_Ownership_Status_Own
                elif home_own == "rent":
                    features[53] = 1.0  # Home_Ownership_Status_Rent
                # Mortgage is default (all 0)
                
                # Investment experience encoding
                inv_exp = user_data.get("Investment_Experience_Level", "").lower()
                if inv_exp == "expert":
                    features[54] = 1.0  # Investment_Experience_Level_Expert
                elif inv_exp == "intermediate":
                    features[55] = 1.0  # Investment_Experience_Level_Intermediate
                # Beginner is default (all 0)
                
                # Financial goals encoding
                fin_goals = user_data.get("Financial_Goals", "").lower()
                if "home" in fin_goals:
                    features[56] = 1.0  # Financial_Goals_Home Purchase
                elif "legacy" in fin_goals:
                    features[57] = 1.0  # Financial_Goals_Legacy Planning
                elif "travel" in fin_goals:
                    features[58] = 1.0  # Financial_Goals_Travel
                # Retirement is default (all 0)
                
                # Insurance coverage encoding
                if user_data.get("Insurance_Coverage", False):
                    features[59] = 1.0  # Insurance_Coverage_Yes
                
                # Tax benefits eligibility encoding
                if user_data.get("Tax_Benefits_Eligibility", False):
                    features[60] = 1.0  # Tax_Benefits_Eligibility_Yes
                
                # Government pension eligibility encoding
                if user_data.get("Government_Pension_Eligibility", False):
                    features[61] = 1.0  # Government_Pension_Eligibility_Yes
                
                # Private pension eligibility encoding
                if user_data.get("Private_Pension_Eligibility", False):
                    features[62] = 1.0  # Private_Pension_Eligibility_Yes
                
                # Pension type encoding
                pension_type = user_data.get("Pension_Type", "").lower()
                if "defined contribution" in pension_type:
                    features[63] = 1.0  # Pension_Type_Defined Contribution
                # Defined Benefit is default (all 0)
                
                # Withdrawal strategy encoding
                withdrawal = user_data.get("Withdrawal_Strategy", "").lower()
                if withdrawal == "dynamic":
                    features[64] = 1.0  # Withdrawal_Strategy_Dynamic
                elif withdrawal == "fixed":
                    features[65] = 1.0  # Withdrawal_Strategy_Fixed
                # Systematic is default (all 0)
                
                # Transaction channel encoding
                trans_channel = user_data.get("Transaction_Channel", "").lower()
                if trans_channel == "branch":
                    features[66] = 1.0  # Transaction_Channel_Branch
                elif trans_channel == "online":
                    features[67] = 1.0  # Transaction_Channel_Online
                # Mobile is default (all 0)
                
                # Previous fraud flag encoding
                if user_data.get("Previous_Fraud_Flag", False):
                    features[68] = 1.0  # Previous_Fraud_Flag_Yes
                
                print(f"✅ Created feature vector with {len(features)} features")
                print(f"📊 Feature summary: {sum(features[:25]):.1f} numeric + {sum(features[25:]):.0f} categorical")
                
                # Scale features if scaler is available
                if scaler is not None:
                    try:
                        features_scaled = scaler.transform([features])
                        print(f"✅ Features scaled using StandardScaler")
                    except Exception as e:
                        print(f"⚠️ Scaling failed: {e}, using unscaled features")
                        features_scaled = [features]
                else:
                    features_scaled = [features]
                
                # Make prediction
                try:
                    prediction = xgb_model.predict(features_scaled)
                    prediction_proba = xgb_model.predict_proba(features_scaled) if hasattr(xgb_model, 'predict_proba') else None
                    print(f"✅ ML prediction successful: {prediction[0]}")
                except Exception as e:
                    print(f"⚠️ ML prediction failed: {e}, falling back to rule-based")
                    raise e
                
                # Determine fraud risk based on prediction
                if prediction[0] == 1:  # Assuming 1 = fraud, 0 = no fraud
                    fraud_risk = "High"
                    fraud_score = 0.9
                else:
                    fraud_risk = "Low"
                    fraud_score = 0.1
                
                # Get probability if available
                if prediction_proba is not None:
                    fraud_score = float(prediction_proba[0][1])  # Probability of fraud class
                    if fraud_score > 0.7:
                        fraud_risk = "High"
                    elif fraud_score > 0.3:
                        fraud_risk = "Medium"
                    else:
                        fraud_risk = "Low"
                
                # Generate suspicious factors based on ML prediction
                suspicious_factors = []
                if fraud_score > 0.5:
                    suspicious_factors.append("ML model detected high fraud probability")
                if user_data["Suspicious_Flag"]:
                    suspicious_factors.append("Transaction flagged as suspicious")
                if user_data["Anomaly_Score"] and user_data["Anomaly_Score"] > 0.8:
                    suspicious_factors.append("High anomaly score detected")
                if user_data["Country"] != user_data["Geo_Location"]:
                    suspicious_factors.append("Geographic location mismatch")
                
                # Generate recommendations
                recommendations = []
                if fraud_risk == "High":
                    recommendations.extend([
                        "Immediately freeze account",
                        "Contact fraud department",
                        "Review recent transactions",
                        "Change security credentials"
                    ])
                elif fraud_risk == "Medium":
                    recommendations.extend([
                        "Monitor account closely",
                        "Enable additional security",
                        "Review transaction patterns"
                    ])
                else:
                    recommendations.extend([
                        "Continue normal monitoring",
                        "Maintain current security measures"
                    ])
                
                return {
                    "fraud_risk": fraud_risk,
                    "fraud_score": round(fraud_score, 3),
                    "suspicious_factors": suspicious_factors,
                    "recommendations": recommendations,
                    "summary": f"ML model analysis: {fraud_risk} risk level with {fraud_score:.1%} fraud probability",
                    "method": "ML Model",
                    "confidence": "High",
                    "model_type": "XGBoost",
                    "features_used": len(features),
                    "data_source": "DATABASE_PENSION_DATA",
                    "note": "This fraud detection analysis is based on the client's pension data stored in our database, not from uploaded documents."
                }
                
        except Exception as e:
            print(f"⚠️ ML model failed: {e}, falling back to rule-based detection")
        
        # FALLBACK: RULE-BASED DETECTION
        print("🔍 Using rule-based fraud detection (fallback)")
        
        prompt = f"""
        **SYSTEM:** You are a Financial Fraud Detection System...
        **TASK:** Analyze the user's transaction data below...
        **FRAUD DETECTION FACTORS:**
        1.  **Geographic Anomaly**: `Geo_Location` doesn't match `Country`.
        2.  **Suspicious Amount**: `Transaction_Amount` is unusually high/low.
        3.  **Flagged Transaction**: `Suspicious_Flag` is True.
        4.  **High Anomaly Score**: `Anomaly_Score` > 0.8.
        **DATA TO ANALYZE:**
        ```json
        {json.dumps(user_data, indent=2)}
        ```
        **OUTPUT INSTRUCTIONS:**
        Return a single JSON object with this structure: {{"fraud_risk": "Low/Medium/High", "fraud_score": float, "suspicious_factors": [], "recommendations": [], "summary": "..."}}
        """
        response = json_llm.invoke(prompt)
        result = json.loads(response.content)
        
        # Add method indicator and data source information
        result["method"] = "Rule-Based"
        result["confidence"] = "Medium"
        result["data_source"] = "DATABASE_PENSION_DATA"
        result["note"] = "This fraud detection analysis is based on the client's pension data stored in our database, not from uploaded documents."
        
        return result
        
    finally:
        db.close()

# --- Tool 3: Pension Projection ---
class ProjectionToolInput(BaseModel):
    user_id: Optional[int] = Field(description="The numeric database ID for the user. If not provided, will be retrieved from current session.")
    query: Optional[str] = Field(description="The user's original query to parse time periods from (e.g., 'retire in 3 years')")

    @validator("user_id", pre=True)
    def coerce_user_id(cls, value):
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        return None

def parse_time_period_from_query(query: str, current_age: int, retirement_age: int) -> int:
    """
    Parse natural language queries to extract the time period for pension calculations.
    Returns the number of years to use in calculations.
    """
    query_lower = query.lower()
    
    # Pattern 1: "retire in X years"
    match = re.search(r'retire\s+in\s+(\d+)\s+years?', query_lower)
    if match:
        years = int(match.group(1))
        print(f"🔍 Time Parsing: Found 'retire in {years} years' → Using {years} years")
        return years
    
    # Pattern 2: "retire at age X"
    match = re.search(r'retire\s+at\s+age\s+(\d+)', query_lower)
    if match:
        target_age = int(match.group(1))
        years = max(0, target_age - current_age)
        print(f"🔍 Time Parsing: Found 'retire at age {target_age}' → Using {years} years")
        return years
    
    # Pattern 3: "retire early" or "retire soon"
    if 'retire early' in query_lower or 'retire soon' in query_lower:
        years = min(5, retirement_age - current_age)  # Assume 5 years or less
        print(f"🔍 Time Parsing: Found 'retire early/soon' → Using {years} years")
        return years
    
    # Pattern 4: "retire next year"
    if 'retire next year' in query_lower:
        years = 1
        print(f"🔍 Time Parsing: Found 'retire next year' → Using {years} years")
        return years
    
    # Pattern 5: "retire in X months"
    match = re.search(r'retire\s+in\s+(\d+)\s+months?', query_lower)
    if match:
        months = int(match.group(1))
        years = months / 12
        print(f"🔍 Time Parsing: Found 'retire in {months} months' → Using {years:.1f} years")
        return max(0.1, years)  # Minimum 0.1 years
    
    # Default: Use full retirement age
    default_years = retirement_age - current_age
    print(f"🔍 Time Parsing: No specific time found → Using default {default_years} years to retirement")
    return default_years

@tool(args_schema=ProjectionToolInput)
def project_pension(user_id: int = None, query: str = None) -> Dict[str, Any]:
    """
    Provides a comprehensive pension overview including current savings, goal progress,
    years remaining, savings rate, and projected balance at retirement.
    Can also calculate specific time-based projections based on the query.
    """
    # PRIORITY 1: Get user_id from tool input (most direct)
    if user_id is None:
        # Try to get from context as fallback
        user_id = get_current_user_id_from_context()
        if user_id:
            print(f"🔍 Context: Using user_id={user_id} from request context")
    
    # PRIORITY 2: Clean up the input if it's not a clean integer
    if user_id is None or isinstance(user_id, str):
        user_id = extract_user_id_from_input(user_id)
        if user_id:
            print(f"🔍 Input Cleanup: Extracted user_id={user_id} from input")
    
    if not user_id:
        return {"error": "User not authenticated. Please log in."}
    
    print(f"🔍 Tool Debug - User ID: {user_id}, Query: {query}")
    
    db: Session = SessionLocal()
    try:
        # Apply role-based context detection using workflow context
        original_query = get_current_query_from_context() or ""
        target_user_id, context_type = detect_role_based_context(original_query, user_id, db)
        if context_type == 'client':
            print(f"🔍 Role Context: Accessing client data (ID: {target_user_id})")
            user_id = target_user_id
        elif context_type == 'self':
            print(f"🔍 Role Context: Accessing own data (ID: {target_user_id})")
            user_id = target_user_id
        else:
            print(f"🔍 Role Context: Unknown context, using original user_id: {user_id}")
        
        pension_data = db.query(models.PensionData).filter(models.PensionData.user_id == user_id).first()
        if not pension_data:
            return {"error": f"No pension data found for User ID: {user_id}"}
        
        # Extract all pension data fields
        current_savings = pension_data.current_savings or 0
        annual_income = pension_data.annual_income or 0
        age = pension_data.age or 0
        retirement_age_goal = pension_data.retirement_age_goal or 65
        annual_contribution = pension_data.contribution_amount or 0
        employer_contribution = pension_data.employer_contribution or 0
        total_annual_contribution = annual_contribution + employer_contribution
        pension_type = pension_data.pension_type or "Defined Contribution"
        # Normalize return rate to ensure it's in decimal format (e.g., 6.88% -> 0.0688)
        raw_return_rate = pension_data.annual_return_rate or 0.08
        if raw_return_rate > 1.0:  # If it's greater than 100%, convert from percentage
            annual_return_rate = raw_return_rate / 100.0
        else:
            annual_return_rate = raw_return_rate
        
        print(f"🔍 Tool Debug - Pension Type: {pension_type}")
        print(f"🔍 Tool Debug - Raw Return Rate: {raw_return_rate}")
        print(f"🔍 Tool Debug - Normalized Return Rate: {annual_return_rate}")
        print(f"🔍 Tool Debug - Current Savings: {current_savings}")
        print(f"🔍 Tool Debug - Annual Income: {annual_income}")
        print(f"🔍 Tool Debug - Age: {age}")
        print(f"🔍 Tool Debug - Retirement Goal: {retirement_age_goal}")
        
        # Parse time period from query if provided
        if query:
            years_to_retirement = parse_time_period_from_query(query, age, retirement_age_goal)
        else:
            years_to_retirement = max(0, retirement_age_goal - age)
        
        print(f"🔍 Tool Debug - Years to Retirement: {years_to_retirement}")
        
        # Calculate retirement goal (example: 10x annual income)
        retirement_goal_amount = annual_income * 10
        
        # Calculate progress
        progress_percentage = min(100, (current_savings / retirement_goal_amount) * 100) if retirement_goal_amount > 0 else 0
        
        # Determine status
        if age >= retirement_age_goal:
            status = "At Retirement Age"
        elif progress_percentage >= 80:
            status = "On Track"
        elif progress_percentage >= 50:
            status = "Good Progress"
        else:
            status = "Needs Attention"
        
        # Calculate savings rate
        savings_rate_percentage = (total_annual_contribution / annual_income) * 100 if annual_income > 0 else 0
        
        # Try to get projections from the projection service first (more accurate)
        try:
            from ..agents.services.projection import run_projection_agent
            
            user_data_for_projection = {
                "current_savings": current_savings,
                "annual_income": annual_income,
                "age": age,
                "retirement_age": retirement_age_goal,
                "annual_contribution": total_annual_contribution,
                "risk_tolerance": pension_data.risk_tolerance,
                "pension_type": pension_type
            }
            
            scenario_params = {
                "inflation_rate": 0.025,  # 2.5% inflation
                "return_rate": annual_return_rate,
                "years": years_to_retirement
            }
            
            print(f"🔍 Projection Service: Calling projection agent with params: {scenario_params}")
            projection_result = run_projection_agent(user_data_for_projection, scenario_params)
            
            # Extract projection data
            projected_balance = projection_result.get("projected_balance", 0)
            nominal_projection = projection_result.get("nominal_projection", 0)
            inflation_adjusted = projection_result.get("inflation_adjusted", True)
            
            print(f"🔍 Projection Service: Successfully got projections - Balance: {projected_balance}, Nominal: {nominal_projection}")
            
        except Exception as e:
            print(f"⚠️ Projection service unavailable: {e}, falling back to basic calculations")
            # Fallback calculations
            projected_balance = current_savings * (1.08 ** years_to_retirement) if years_to_retirement > 0 else current_savings
            nominal_projection = projected_balance
            inflation_adjusted = False
        
        # Calculate projections based on pension type with realistic limits (fallback)
        if pension_type.lower() in ["defined contribution", "dc", "defined contribution plan"]:
            # DEFINED CONTRIBUTION: Future value based on contributions + investment returns
            if years_to_retirement > 0:
                # Use more conservative return rate for longer periods
                if years_to_retirement > 20:
                    effective_return_rate = min(annual_return_rate, 0.06)  # Cap at 6% for very long periods
                elif years_to_retirement > 10:
                    effective_return_rate = min(annual_return_rate, 0.07)  # Cap at 7% for long periods
                else:
                    effective_return_rate = annual_return_rate
                
                # Future value of current savings
                fv_current = current_savings * (1 + effective_return_rate) ** years_to_retirement
                
                # Future value of contributions (more conservative)
                if total_annual_contribution > 0:
                    fv_contributions = total_annual_contribution * ((1 + effective_return_rate) ** years_to_retirement - 1) / effective_return_rate
                else:
                    fv_contributions = 0
                
                projected_balance = fv_current + fv_contributions
                
                # Apply realistic limits to prevent impossible projections
                max_reasonable_multiplier = min(10, years_to_retirement * 0.5)  # More conservative
                max_reasonable_projection = current_savings * max_reasonable_multiplier
                
                if projected_balance > max_reasonable_projection:
                    projected_balance = max_reasonable_projection
                    print(f"🔍 Tool Debug - Capped projection from £{projected_balance:,.0f} to £{max_reasonable_projection:,.0f}")
                
                # Calculate scenarios with realistic limits
                scenario_10_percent = min(projected_balance * 1.1, max_reasonable_projection * 1.1)
                scenario_20_percent = min(projected_balance * 1.2, max_reasonable_projection * 1.2)
                
                print(f"🔍 Tool Debug - DC Calculation: FV Current=£{fv_current:,.0f}, FV Contributions=£{fv_contributions:,.0f}")
                print(f"🔍 Tool Debug - Projected Balance: £{projected_balance:,.0f}")
                
            else:
                projected_balance = current_savings
                scenario_10_percent = current_savings
                scenario_20_percent = current_savings
                
        elif pension_type.lower() in ["defined benefit", "db", "defined benefit plan"]:
            # DEFINED BENEFIT: Pension based on salary, years of service, and benefit formula
            # For DB plans, contributions don't directly affect the benefit
            projected_balance = pension_data.projected_pension_amount or (annual_income * 0.6)  # 60% of final salary
            scenario_10_percent = projected_balance
            scenario_20_percent = projected_balance
            print(f"🔍 Tool Debug - DB Plan: Using projected amount £{projected_balance:,.0f}")
            
        else:
            # HYBRID or UNKNOWN: Use a conservative approach
            if years_to_retirement > 0:
                # More conservative return rate
                conservative_return = min(annual_return_rate * 0.8, 0.06)  # 80% of original rate, max 6%
                projected_balance = current_savings * (1 + conservative_return) ** years_to_retirement
                
                # Apply realistic limits
                max_reasonable = current_savings * min(8, years_to_retirement * 0.4)
                if projected_balance > max_reasonable:
                    projected_balance = max_reasonable
                
                scenario_10_percent = min(projected_balance * 1.1, max_reasonable * 1.1)
                scenario_20_percent = min(projected_balance * 1.2, max_reasonable * 1.2)
            else:
                projected_balance = current_savings
                scenario_10_percent = current_savings
                scenario_20_percent = current_savings
            
            print(f"🔍 Tool Debug - Hybrid/Unknown: Conservative calculation £{projected_balance:,.0f}")
        
        # Validation and warnings
        validation_warnings = []
        calculation_notes = []
        
        # Check for unrealistic projections
        if projected_balance > current_savings * 20:
            validation_warnings.append("Projection may be optimistic - consider reviewing assumptions")
            calculation_notes.append("Large projection due to long time horizon or high return assumptions")
        
        if years_to_retirement < 1:
            calculation_notes.append("User is at or past retirement age - projection shows current status")
        
        # Additional validation for very short time periods
        if years_to_retirement <= 3 and projected_balance > current_savings * 2:
            validation_warnings.append("Short-term projection seems high - consider more conservative estimates")
            calculation_notes.append("Short time periods typically show smaller growth")
        
        # Generate chart data for visualizations
        chart_data = {}
        
        # Chart 1: Pension Growth Over Time
        if years_to_retirement > 0:
            years_data = list(range(age, retirement_age_goal + 1))
            projected_values = []
            for year in years_data:
                years_from_now = year - age
                if years_from_now <= 0:
                    projected_values.append(current_savings)
                else:
                    # Calculate projected value for this year
                    fv_current = current_savings * (1 + annual_return_rate) ** years_from_now
                    if total_annual_contribution > 0:
                        fv_contributions = total_annual_contribution * ((1 + annual_return_rate) ** years_from_now - 1) / annual_return_rate
                    else:
                        fv_contributions = 0
                    projected_values.append(min(fv_current + fv_contributions, current_savings * 20))  # Cap at 20x
            
            chart_data["pension_growth"] = {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "description": "Pension growth projection over time",
                "data": {
                    "values": [{"age": y, "projected_value": v} for y, v in zip(years_data, projected_values)]
                },
                "mark": "line",
                "encoding": {
                    "x": {
                        "field": "age",
                        "type": "quantitative",
                        "title": "Age"
                    },
                    "y": {
                        "field": "projected_value",
                        "type": "quantitative",
                        "title": "Projected Pension Value (£)"
                    }
                }
            }
        
        # Chart 2: Progress to Goal
        chart_data["progress_to_goal"] = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "Progress toward retirement goal",
            "data": {
                "values": [
                    {"metric": "Current Savings", "value": current_savings},
                    {"metric": "Goal Amount", "value": retirement_goal_amount}
                ]
            },
            "mark": "bar",
            "encoding": {
                "x": {
                    "field": "metric",
                    "type": "nominal",
                    "title": ""
                },
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "title": "Amount (£)"
                }
            }
        }
        
        # Chart 3: Savings Rate Analysis
        chart_data["savings_analysis"] = {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "description": "Savings rate and contribution analysis",
            "data": {
                "values": [
                    {"category": "Annual Income", "amount": annual_income},
                    {"category": "Annual Contribution", "amount": total_annual_contribution},
                    {"category": "Current Savings", "amount": current_savings}
                ]
            },
            "mark": "bar",
            "encoding": {
                "x": {
                    "field": "category",
                    "type": "nominal",
                    "title": ""
                },
                "y": {
                    "field": "amount",
                    "type": "quantitative",
                    "title": "Amount (£)"
                }
            }
        }
        
        # Format the response
        response = {
            "current_data": {
                "current_savings": f"£{current_savings:,.0f}",
                "annual_income": f"£{annual_income:,.0f}",
                "age": age,
                "retirement_age_goal": retirement_age_goal,
                "annual_contribution": f"£{total_annual_contribution:,.0f}",
                "savings_rate": f"{savings_rate_percentage:.1f}%",
                "pension_type": pension_type
            },
            "projection_analysis": {
                "years_to_retirement": years_to_retirement,
                "projected_balance": f"£{projected_balance:,.0f}",
                "scenario_10_percent_increase": f"£{scenario_10_percent:,.0f}",
                "scenario_20_percent_increase": f"£{scenario_20_percent:,.0f}",
                "annual_return_rate": f"{annual_return_rate * 100:.1f}%",
                "validation_warnings": validation_warnings,
                "calculation_notes": calculation_notes
            },
            "status": status,
            "progress_to_goal": f"{progress_percentage:.1f}%",
            "chart_data": chart_data,
            "data_source": "DATABASE_PENSION_DATA",
            "note": f"This analysis is based on the {'client' if context_type == 'client' else 'user' if context_type == 'self' else 'user'}'s pension data stored in our database, not from uploaded documents."
        }
        
        return response
        
    except Exception as e:
        print(f"🔍 Tool Error: {str(e)}")
        return {"error": f"Error processing pension projection: {str(e)}"}
    finally:
        db.close()

# --- Tool 4: Knowledge Base Search (Enhanced) ---
class KnowledgeSearchInput(BaseModel):
    query: str = Field(description="The search query for the knowledge base.")
    user_id: Optional[int] = Field(description="The numeric database ID for the user. If not provided, will be retrieved from current session.")

    @validator("user_id", pre=True)
    def coerce_user_id(cls, value):
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        return None

@tool(args_schema=KnowledgeSearchInput)
def knowledge_base_search(query: str, user_id: int = None) -> Dict[str, Any]:
    """
    Searches the knowledge base for relevant information about pensions, retirement planning,
    and financial advice. Returns structured information based on the query.
    This tool searches BOTH general pension knowledge AND the user's uploaded documents.
    """
    # PRIORITY 1: Get user_id from request context (most secure)
    if user_id is None:
        user_id = get_current_user_id_from_context()
        if user_id:
            print(f"🔍 Context: Using user_id={user_id} from request context")
    
    # PRIORITY 2: Clean up the input if it's not a clean integer
    if user_id is None or isinstance(user_id, str):
        user_id = extract_user_id_from_input(user_id)
        if user_id:
            print(f"🔍 Input Cleanup: Extracted user_id={user_id} from input")
    
    if not user_id:
        return {"error": "User not authenticated. Please log in."}
    
    print(f"\n--- TOOL: Searching Knowledge Base for User ID: {user_id} ---")
    
    all_results = []
    
    try:
        # Get database session for role-based context
        db: Session = SessionLocal()
        # Apply role-based context detection
        target_user_id, context_type = detect_role_based_context(query, user_id, db)
        if context_type == 'client':
            print(f"🔍 Role Context: Accessing client data (ID: {target_user_id})")
            user_id = target_user_id
        elif context_type == 'unknown':
            print(f"🔍 Role Context: Unknown context, using original user_id: {user_id}")
        # SEARCH 1: General pension knowledge base
        print(f"🔍 Searching general pension knowledge base...")
        general_collection = get_or_create_collection("pension_knowledge")
        general_results = query_collection(general_collection, [query], n_results=2)
        
        # Handle ChromaDB results properly
        if general_results and isinstance(general_results, dict) and 'documents' in general_results:
            general_docs = general_results.get('documents', [])
            general_metadatas = general_results.get('metadatas', [])
            general_distances = general_results.get('distances', [])
            
            # ChromaDB returns nested lists, flatten them
            if general_docs and isinstance(general_docs[0], list):
                general_docs = general_docs[0]
            if general_metadatas and isinstance(general_metadatas[0], list):
                general_metadatas = general_metadatas[0]
            if general_distances and isinstance(general_distances[0], list):
                general_distances = general_distances[0]
            
            for i, (doc, metadata, distance) in enumerate(zip(general_docs, general_metadatas, general_distances)):
                if doc:  # Only add if document content exists
                    all_results.append({
                        "result": len(all_results) + 1,
                        "content": doc,
                        "source": "General Pension Knowledge Base",
                        "type": "general_knowledge",
                        "relevance_score": 1 - distance if isinstance(distance, (int, float)) else 0.5
                    })
        
        # SEARCH 2: FAQ Knowledge Base
        print(f"🔍 Searching FAQ knowledge base...")
        faq_collection = get_or_create_collection("faq_collection")
        faq_results = query_collection(faq_collection, [query], n_results=3)
        
        # Handle FAQ ChromaDB results properly
        if faq_results and isinstance(faq_results, dict) and 'documents' in faq_results:
            faq_docs = faq_results.get('documents', [])
            faq_metadatas = faq_results.get('metadatas', [])
            faq_distances = faq_results.get('distances', [])
            
            # ChromaDB returns nested lists, flatten them
            if faq_docs and isinstance(faq_docs[0], list):
                faq_docs = faq_docs[0]
            if faq_metadatas and isinstance(faq_metadatas[0], list):
                faq_metadatas = faq_metadatas[0]
            if faq_distances and isinstance(faq_distances[0], list):
                faq_distances = faq_distances[0]
            
            for i, (doc, metadata, distance) in enumerate(zip(faq_docs, faq_metadatas, faq_distances)):
                if doc:  # Only add if document content exists
                    # Extract question and answer from metadata
                    question = metadata.get('question', '') if metadata else ''
                    answer = metadata.get('answer', '') if metadata else ''
                    
                    # Create a formatted FAQ result
                    faq_content = f"Question: {question}\nAnswer: {answer}" if question and answer else doc
                    
                    all_results.append({
                        "result": len(all_results) + 1,
                        "content": faq_content,
                        "source": "FAQ Knowledge Base",
                        "type": "faq_knowledge",
                        "relevance_score": 1 - distance if isinstance(distance, (int, float)) else 0.5
                    })
        
        # SEARCH 3: User's uploaded PDF documents
        print(f"🔍 Searching user's uploaded documents...")
        user_docs_collection = get_or_create_collection(f"user_{user_id}_docs")
        user_results = query_collection(user_docs_collection, [query], n_results=3)
        
        # Handle ChromaDB results properly
        if user_results:
            if isinstance(user_results, dict) and 'documents' in user_results:
                user_docs = user_results.get('documents', [])
                user_metadatas = user_results.get('metadatas', [])
                user_distances = user_results.get('distances', [])
                
                # ChromaDB returns nested lists, flatten them
                if user_docs and isinstance(user_docs[0], list):
                    user_docs = user_docs[0]
                if user_metadatas and isinstance(user_metadatas[0], list):
                    user_metadatas = user_metadatas[0]
                if user_distances and isinstance(user_distances[0], list):
                    user_distances = user_distances[0]
                    
            elif isinstance(user_results, list):
                # If it's a list, assume it's the documents directly
                user_docs = user_results
                user_metadatas = [{}] * len(user_results)
                user_distances = [0.0] * len(user_results)
            else:
                user_docs = []
                user_metadatas = []
                user_distances = []
            
            for i, (doc, metadata, distance) in enumerate(zip(user_docs, user_metadatas, user_distances)):
                if doc:  # Only add if document content exists
                    # Handle distance calculation safely
                    try:
                        if isinstance(distance, (int, float)):
                            relevance_score = 1 - distance
                        elif isinstance(distance, list) and len(distance) > 0:
                            first_distance = distance[0]
                            if isinstance(first_distance, (int, float)):
                                relevance_score = 1 - first_distance
                            else:
                                relevance_score = 0.5
                        else:
                            relevance_score = 0.5
                    except (TypeError, ValueError):
                        relevance_score = 0.5
                    
                    all_results.append({
                        "result": len(all_results) + 1,
                        "content": doc,
                        "source": metadata.get('source', f'User {user_id} Document') if metadata else f'User {user_id} Document',
                        "type": "user_document",
                        "relevance_score": relevance_score
                    })
        
        # If no results found anywhere
        if not all_results:
            return {
                "found": False,
                "message": "No relevant information found in the general knowledge base or your uploaded documents.",
                "suggestions": [
                    "Try rephrasing your question",
                    "Use more specific terms",
                    "Check if your question is related to pensions, retirement, or financial planning",
                    "Make sure you have uploaded relevant documents"
                ]
            }
        
        # Sort results by relevance score
        all_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Update result numbers after sorting
        for i, result in enumerate(all_results):
            result["result"] = i + 1
        
        return {
            "found": True,
            "query": query,
            "results": all_results,
            "total_results": len(all_results),
            "search_summary": {
                "general_knowledge_results": len([r for r in all_results if r.get('type') == 'general_knowledge']),
                "user_document_results": len([r for r in all_results if r.get('type') == 'user_document']),
                "total_found": len(all_results)
            },
            "summary": f"Found {len(all_results)} relevant results for your query about '{query}'. "
                      f"This includes both general pension knowledge and information from your uploaded documents."
        }
        
    except Exception as e:
        return {"error": f"Error searching knowledge base: {str(e)}"}

# --- Tool 5: Document Analysis ---
class DocumentAnalysisInput(BaseModel):
    query: str = Field(description="The question or query about the uploaded document")
    user_id: Optional[int] = Field(description="User ID to identify which documents to search. If not provided, will be retrieved from context.")

@tool(args_schema=DocumentAnalysisInput)
def analyze_uploaded_document(query: str, user_id: int = None) -> Dict[str, Any]:
    """
    Analyzes the user's uploaded PDF documents to answer specific questions.
    This tool searches through all documents uploaded by the user and provides
    relevant information based on the query.
    """
    # PRIORITY 1: Get user_id from request context (most secure)
    if user_id is None:
        user_id = get_current_user_id_from_context()
        if user_id:
            print(f"🔍 Context: Using user_id={user_id} from request context")
    
    # PRIORITY 2: Clean up the input if it's not a clean integer
    if user_id is None or isinstance(user_id, str):
        user_id = extract_user_id_from_input(user_id)
        if user_id:
            print(f"🔍 Input Cleanup: Extracted user_id={user_id} from input")
    
    if not user_id:
        return {"error": "User not authenticated. Please log in."}
    
    print(f"\n--- TOOL: Analyzing Uploaded Documents for User ID: {user_id} ---")
    
    try:
        # Apply role-based context detection
        db: Session = SessionLocal()
        target_user_id, context_type = detect_role_based_context(query, user_id, db)
        if context_type == 'client':
            print(f"🔍 Role Context: Accessing client data (ID: {target_user_id})")
            user_id = target_user_id
        elif context_type == 'unknown':
            print(f"🔍 Role Context: Unknown context, using original user_id: {user_id}")
        
        # Search user's uploaded documents
        user_docs_collection = get_or_create_collection(f"user_{user_id}_docs")
        user_results = query_collection(user_docs_collection, [query], n_results=5)
        
        # ChromaDB returns results in different formats, handle both
        if isinstance(user_results, dict):
            user_docs = user_results.get('documents', [])
            user_metadatas = user_results.get('metadatas', [])
            user_distances = user_results.get('distances', [])
            
            # ChromaDB returns nested lists, flatten them
            if user_docs and isinstance(user_docs[0], list):
                user_docs = user_docs[0]
            if user_metadatas and isinstance(user_metadatas[0], list):
                user_metadatas = user_metadatas[0]
            if user_distances and isinstance(user_distances[0], list):
                user_distances = user_distances[0]
                
        elif isinstance(user_results, list):
            # If it's a list, assume it's the documents directly
            user_docs = user_results
            user_metadatas = [{}] * len(user_results)
            user_distances = [0.0] * len(user_results)
        else:
            return {
                "found": False,
                "message": f"Unexpected result format from document search: {type(user_results)}",
                "suggestions": ["Try again or contact support"]
            }
        
        if not user_docs:
            return {
                "found": False,
                "message": "No uploaded documents found for analysis.",
                "suggestions": [
                    "Upload a PDF document first using the upload endpoint",
                    "Check if your document was successfully processed",
                    "Try a different query related to your uploaded content"
                ]
            }
        
        # Format the results
        formatted_results = []
        for i, (doc, metadata, distance) in enumerate(zip(user_docs, user_metadatas, user_distances)):
            if doc:  # Only add if document content exists
                # Handle distance calculation safely
                try:
                    if isinstance(distance, (int, float)):
                        relevance_score = 1 - distance
                    elif isinstance(distance, list) and len(distance) > 0:
                        first_distance = distance[0]
                        if isinstance(first_distance, (int, float)):
                            relevance_score = 1 - first_distance
                        else:
                            relevance_score = 0.5
                    else:
                        relevance_score = 0.5
                except (TypeError, ValueError):
                    relevance_score = 0.5
                
                formatted_results.append({
                    "result": i + 1,
                    "content": doc,
                    "source": metadata.get('source', f'User {user_id} Document') if metadata else f'User {user_id} Document',
                    "relevance_score": relevance_score,
                    "analysis": f"Document chunk {i+1} contains relevant information for your query: '{query}'"
                })
        
        # Sort by relevance
        formatted_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # Update result numbers after sorting
        for i, result in enumerate(formatted_results):
            result["result"] = i + 1
        
        return {
            "found": True,
            "query": query,
            "document_analysis": True,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "summary": f"Successfully analyzed your uploaded documents for '{query}'. "
                      f"Found {len(formatted_results)} relevant document sections.",
            "capabilities": [
                "Search through all your uploaded PDF documents",
                "Extract relevant information based on your questions",
                "Provide context-aware answers from your documents",
                "Support for multiple document types and formats"
            ]
        }
        
    except Exception as e:
        return {"error": f"Error analyzing uploaded documents: {str(e)}"}

# --- Tool 6: Enhanced PDF Document Search ---
class PDFSearchInput(BaseModel):
    query: str = Field(description="The search query for the PDF documents.")
    user_id: Optional[int] = Field(description="The numeric database ID for the user. If not provided, will be retrieved from current session.")

    @validator("user_id", pre=True)
    def coerce_user_id(cls, value):
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group(0))
        return None

@tool(args_schema=PDFSearchInput)
def query_knowledge_base(query: str, user_id: int = None) -> Dict[str, Any]:
    """
    Searches the user's uploaded PDF documents and knowledge base for relevant information.
    Returns structured information based on the query.
    """
    # PRIORITY 1: Get user_id from tool input (most direct)
    if user_id is None:
        # Try to get from context as fallback
        user_id = get_current_user_id_from_context()
        if user_id:
            print(f"🔍 Context: Using user_id={user_id} from request context")
    
    # PRIORITY 2: Clean up the input if it's not a clean integer
    if user_id is None or isinstance(user_id, str):
        user_id = extract_user_id_from_input(user_id)
        if user_id:
            print(f"🔍 Input Cleanup: Extracted user_id={user_id} from input")
    
    if not user_id:
        return {"error": "User not authenticated. Please log in."}
    
    print(f"\n--- TOOL: Searching Knowledge Base for User ID: {user_id} ---")
    
    # 🔍 FIX: Extract actual query from JSON input if needed
    actual_query = query
    if query.startswith('{') and query.endswith('}'):
        try:
            import json
            query_data = json.loads(query)
            if 'query' in query_data:
                actual_query = query_data['query']
                print(f"🔍 Fixed: Extracted actual query: '{actual_query}' from JSON input")
        except:
            print(f"🔍 Warning: Could not parse JSON input, using as-is")
    
    try:
        # Apply role-based context detection
        db: Session = SessionLocal()
        target_user_id, context_type = detect_role_based_context(actual_query, user_id, db)
        if context_type == 'client':
            print(f"🔍 Role Context: Accessing client data (ID: {target_user_id})")
            user_id = target_user_id
        elif context_type == 'unknown':
            print(f"🔍 Role Context: Unknown context, using original user_id: {user_id}")
        
        # Get the user's document collection
        collection_name = f"user_{user_id}_docs"
        collection = get_or_create_collection(collection_name)
        
        # 🔍 DEBUG: Check what we're passing to query_collection
        print(f"🔍 Debug: actual_query = '{actual_query}' (type: {type(actual_query)})")
        print(f"🔍 Debug: user_id = {user_id} (type: {type(user_id)})")
        
        # Search the collection - ensure query is a string
        if isinstance(actual_query, str):
            query_texts = [actual_query]
        elif isinstance(actual_query, list):
            query_texts = actual_query
        else:
            query_texts = [str(actual_query)]
        
        print(f"🔍 Debug: query_texts = {query_texts}")
        
        results = query_collection(collection, query_texts, n_results=3)
        
        # 🔍 DEBUG: Check what results we got back
        print(f"🔍 Debug: results type = {type(results)}")
        print(f"🔍 Debug: results keys = {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
        
        if not results or not isinstance(results, dict) or not results.get('documents'):
            return {
                "found": False,
                "message": "No relevant information found in your uploaded documents.",
                "suggestions": [
                    "Try rephrasing your question",
                    "Use more specific terms",
                    "Check if you have uploaded relevant PDF documents",
                    "Make sure your question is related to pension, retirement, or financial planning"
                ],
                "user_id": user_id,
                "query": actual_query,
                "search_type": "PDF_DOCUMENT_SEARCH",
                "pdf_status": "NO_PDFS_FOUND",
                "note": "This response is from searching your uploaded PDF documents. No relevant documents were found for your query."
            }
        
        # Format the results
        formatted_results = []
        documents = results.get('documents', [])
        metadatas = results.get('metadatas', [])
        distances = results.get('distances', [])
        
        for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
            # 🔍 FIX: Handle OCR placeholder content gracefully
            content = doc
            if isinstance(content, list):
                content = ' '.join(str(item) for item in content)
            
            # Check if content is just OCR placeholder
            if 'OCR processing required' in str(content) or 'Scanned content detected' in str(content):
                content = "This document appears to be scanned images. The content is not searchable as text. Please upload a text-based PDF or contact support for OCR processing."
            
            formatted_results.append({
                "result": i + 1,
                "content": content,
                "source": metadata.get('source', 'Unknown PDF') if isinstance(metadata, dict) else 'Unknown PDF',
                "relevance_score": round(1 - distance, 3) if isinstance(distance, (int, float)) else 0.0,
                "chunk_index": metadata.get('chunk_index', 'Unknown') if isinstance(metadata, dict) else 'Unknown'
            })
        
        return {
            "found": True,
            "query": actual_query,
            "results": formatted_results,
            "total_results": len(formatted_results),
            "summary": f"Found {len(formatted_results)} relevant sections from your uploaded documents for your query about '{actual_query}'.",
            "user_id": user_id,
            "search_scope": f"User {user_id}'s PDF documents",
            "search_type": "PDF_DOCUMENT_SEARCH",
            "pdf_status": "PDFS_FOUND_AND_SEARCHED",
            "note": "This response is based on content extracted from your uploaded PDF documents.",
            "recommendations": [
                "The information above is extracted from your uploaded PDF documents",
                "For the most accurate answers, refer to the original document sources",
                "Consider consulting with a financial advisor for personalized advice"
            ]
        }
        
    except Exception as e:
        print(f"❌ Error searching knowledge base: {e}")
        import traceback
        traceback.print_exc()  # Print full error traceback
        return {
            "error": f"Error searching knowledge base: {str(e)}",
            "user_id": user_id,
            "query": actual_query,
            "search_type": "PDF_DOCUMENT_SEARCH",
            "pdf_status": "ERROR_OCCURRED",
            "note": "An error occurred while searching your PDF documents."
        }

# --- Tool 7: Regulator System Analysis (NEW) ---
@tool
def analyze_system_wide_risk() -> Dict[str, Any]:
    """
    Analyzes risk across ALL users in the system for regulatory oversight.
    This tool is only available to regulators and provides system-wide risk assessment.
    """
    # Check if current user is a regulator
    current_user_id = get_current_user_id_from_context()
    if not current_user_id:
        return {"error": "User not authenticated. Please log in."}
    
    db: Session = SessionLocal()
    try:
        # Verify user is a regulator
        current_user = db.query(models.User).filter(models.User.id == current_user_id).first()
        if not current_user or current_user.role != 'regulator':
            return {"error": "This tool is only available to regulators"}
        
        print(f"🔍 Regulator Tool: Analyzing system-wide risk for regulator {current_user_id}")
        
        # Get all users with pension data
        all_users = db.query(models.PensionData).all()
        
        if not all_users:
            return {"error": "No pension data found in the system"}
        
        # Analyze risk distribution
        risk_levels = {"Low": 0, "Medium": 0, "High": 0}
        total_users = len(all_users)
        total_age = 0
        total_income = 0
        total_savings = 0
        high_risk_users = []
        
        for user_data in all_users:
            # Calculate risk score (simplified version)
            risk_score = 0
            if user_data.volatility and user_data.volatility > 3.5:
                risk_score += 1
            if user_data.portfolio_diversity_score and user_data.portfolio_diversity_score < 0.5:
                risk_score += 1
            if user_data.debt_level and user_data.annual_income and user_data.debt_level > (user_data.annual_income * 0.5):
                risk_score += 1
            if user_data.health_status == 'Poor':
                risk_score += 1
            
            # Categorize risk
            if risk_score <= 1:
                risk_level = "Low"
            elif risk_score <= 2:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            risk_levels[risk_level] += 1
            
            # Collect stats
            if user_data.age:
                total_age += user_data.age
            if user_data.annual_income:
                total_income += user_data.annual_income
            if user_data.current_savings:
                total_savings += user_data.current_savings
            
            # Track high-risk users
            if risk_level == "High":
                high_risk_users.append({
                    "user_id": user_data.user_id,
                    "risk_score": risk_score,
                    "volatility": user_data.volatility,
                    "diversity_score": user_data.portfolio_diversity_score,
                    "debt_ratio": (user_data.debt_level / user_data.annual_income * 100) if user_data.debt_level and user_data.annual_income else 0
                })
        
        # Calculate averages
        avg_age = total_age / total_users if total_users > 0 else 0
        avg_income = total_income / total_users if total_users > 0 else 0
        avg_savings = total_savings / total_users if total_users > 0 else 0
        
        return {
            "system_analysis": True,
            "total_users": total_users,
            "risk_distribution": risk_levels,
            "high_risk_count": len(high_risk_users),
            "high_risk_users": high_risk_users[:10],  # Top 10 high-risk users
            "averages": {
                "age": round(avg_age, 1),
                "income": f"£{avg_income:,.0f}",
                "savings": f"£{avg_savings:,.0f}"
            },
            "data_source": "SYSTEM_WIDE_ANALYSIS",
            "note": "This analysis covers all users in the system for regulatory oversight."
        }
        
    finally:
        db.close()

@tool
def analyze_system_wide_fraud() -> Dict[str, Any]:
    """
    Analyzes fraud patterns across ALL users in the system for regulatory oversight.
    This tool is only available to regulators and provides system-wide fraud detection.
    """
    # Check if current user is a regulator
    current_user_id = get_current_user_id_from_context()
    if not current_user_id:
        return {"error": "User not authenticated. Please log in."}
    
    db: Session = SessionLocal()
    try:
        # Verify user is a regulator
        current_user = db.query(models.User).filter(models.User.id == current_user_id).first()
        if not current_user or current_user.role != 'regulator':
            return {"error": "This tool is only available to regulators"}
        
        print(f"🔍 Regulator Tool: Analyzing system-wide fraud for regulator {current_user_id}")
        
        # Get all users with pension data
        all_users = db.query(models.PensionData).all()
        
        if not all_users:
            return {"error": "No pension data found in the system"}
        
        # Analyze fraud patterns
        suspicious_transactions = 0
        high_anomaly_users = 0
        geographic_anomalies = 0
        fraud_risk_summary = {"high": 0, "medium": 0, "low": 0}
        
        for user_data in all_users:
            # Check suspicious flags
            if user_data.suspicious_flag:
                suspicious_transactions += 1
            
            # Check anomaly scores
            if user_data.anomaly_score and user_data.anomaly_score > 0.8:
                high_anomaly_users += 1
            
            # Categorize fraud risk
            risk_score = 0
            if user_data.suspicious_flag:
                risk_score += 2
            if user_data.anomaly_score and user_data.anomaly_score > 0.8:
                risk_score += 2
            elif user_data.anomaly_score and user_data.anomaly_score > 0.5:
                risk_score += 1
            
            if risk_score >= 3:
                fraud_risk_summary["high"] += 1
            elif risk_score >= 1:
                fraud_risk_summary["medium"] += 1
            else:
                fraud_risk_summary["low"] += 1
        
        return {
            "system_analysis": True,
            "total_users": len(all_users),
            "fraud_risk_summary": fraud_risk_summary,
            "suspicious_transactions": suspicious_transactions,
            "high_anomaly_users": high_anomaly_users,
            "data_source": "SYSTEM_WIDE_FRAUD_ANALYSIS",
            "note": "This fraud analysis covers all users in the system for regulatory oversight."
        }
        
    finally:
        db.close()

@tool
def analyze_geographic_risk() -> Dict[str, Any]:
    """
    Analyzes geographic risk patterns across ALL users in the system for regulatory oversight.
    This tool is only available to regulators and provides system-wide geographic analysis.
    """
    # Check if current user is a regulator
    current_user_id = get_current_user_id_from_context()
    if not current_user_id:
        return {"error": "User not authenticated. Please log in."}
    
    db: Session = SessionLocal()
    try:
        # Verify user is a regulator
        current_user = db.query(models.User).filter(models.User.id == current_user_id).first()
        if not current_user or current_user.role != 'regulator':
            return {"error": "This tool is only available to regulators"}
        
        print(f"🔍 Regulator Tool: Analyzing geographic risk for regulator {current_user_id}")
        
        # Get all users with pension data
        all_users = db.query(models.PensionData).all()
        
        if not all_users:
            return {"error": "No pension data found in the system"}
        
        # Define geographic risk factors for different countries
        country_risk_factors = {
            "UK": {"currency_stability": "High", "regulatory_stability": "High", "economic_risk": "Low"},
            "USA": {"currency_stability": "High", "regulatory_stability": "Medium", "economic_risk": "Low"},
            "Canada": {"currency_stability": "High", "regulatory_stability": "High", "economic_risk": "Low"},
            "Australia": {"currency_stability": "Medium", "regulatory_stability": "High", "economic_risk": "Low"},
            "Germany": {"currency_stability": "High", "regulatory_stability": "High", "economic_risk": "Low"},
        }
        
        # Analyze geographic patterns
        countries = {}
        geographic_anomalies = 0
        suspicious_locations = 0
        cross_border_transactions = 0
        
        for user_data in all_users:
            country = user_data.country or "Unknown"
            if country not in countries:
                risk_info = country_risk_factors.get(country, {
                    "currency_stability": "Unknown", 
                    "regulatory_stability": "Unknown", 
                    "economic_risk": "Unknown"
                })
                countries[country] = {
                    "count": 0,
                    "total_income": 0,
                    "total_savings": 0,
                    "avg_risk_score": 0,
                    "high_value_accounts": 0,
                    "currency_stability": risk_info["currency_stability"],
                    "regulatory_stability": risk_info["regulatory_stability"],
                    "economic_risk": risk_info["economic_risk"]
                }
            
            countries[country]["count"] += 1
            if user_data.annual_income:
                countries[country]["total_income"] += user_data.annual_income
            if user_data.current_savings:
                countries[country]["total_savings"] += user_data.current_savings
            
            # Check for high-value accounts (potential geographic concentration risk)
            if user_data.current_savings and user_data.current_savings > 500000:  # £500k+
                countries[country]["high_value_accounts"] += 1
            
            # Check for geographic anomalies (more realistic)
            if user_data.ip_address and user_data.country:
                # Simulate IP geolocation check (in real system, you'd use IP geolocation service)
                # For now, flag if transaction patterns are unusual
                if user_data.transaction_pattern_score and user_data.transaction_pattern_score > 0.8:
                    suspicious_locations += 1
            
            # Check for potential cross-border transaction risks
            if user_data.transaction_channel and "international" in user_data.transaction_channel.lower():
                cross_border_transactions += 1
        
        # Calculate averages and risk scores for each country
        total_assets_by_country = {}
        for country, data in countries.items():
            if data["count"] > 0:
                data["avg_income"] = data["total_income"] / data["count"]
                data["avg_savings"] = data["total_savings"] / data["count"]
                
                # Calculate geographic concentration risk
                total_country_assets = data["total_savings"]
                total_assets_by_country[country] = total_country_assets
                
                # Assess country-specific risk level
                risk_score = 0
                if data["currency_stability"] == "Low":
                    risk_score += 2
                elif data["currency_stability"] == "Medium":
                    risk_score += 1
                
                if data["regulatory_stability"] == "Low":
                    risk_score += 2
                elif data["regulatory_stability"] == "Medium":
                    risk_score += 1
                
                if data["economic_risk"] == "High":
                    risk_score += 2
                elif data["economic_risk"] == "Medium":
                    risk_score += 1
                
                # Concentration risk (if one country has >40% of total assets)
                data["risk_level"] = "Low" if risk_score <= 1 else "Medium" if risk_score <= 3 else "High"
        
        # Calculate total assets across all countries for concentration analysis
        total_system_assets = sum(total_assets_by_country.values())
        geographic_concentration_risks = []
        
        for country, assets in total_assets_by_country.items():
            if total_system_assets > 0:
                concentration_percentage = (assets / total_system_assets) * 100
                if concentration_percentage > 40:  # More than 40% in one country
                    geographic_concentration_risks.append({
                        "country": country,
                        "concentration_percentage": round(concentration_percentage, 2),
                        "risk_level": "High"
                    })
                elif concentration_percentage > 25:  # More than 25% in one country
                    geographic_concentration_risks.append({
                        "country": country,
                        "concentration_percentage": round(concentration_percentage, 2),
                        "risk_level": "Medium"
                    })
        
        return {
            "system_analysis": True,
            "total_users": len(all_users),
            "countries": countries,
            "geographic_risk_summary": {
                "suspicious_locations": suspicious_locations,
                "cross_border_transactions": cross_border_transactions,
                "concentration_risks": geographic_concentration_risks,
                "total_system_assets": f"£{total_system_assets:,.0f}"
            },
            "key_findings": [
                f"Analyzed {len(all_users)} users across {len(countries)} countries",
                f"Identified {len(geographic_concentration_risks)} countries with high asset concentration",
                f"Found {suspicious_locations} accounts with suspicious location patterns",
                f"Detected {cross_border_transactions} potential cross-border transaction risks"
            ],
            "data_source": "SYSTEM_WIDE_GEOGRAPHIC_ANALYSIS",
            "note": "This analysis evaluates geographic distribution, concentration risks, and location-based anomalies for regulatory oversight."
        }
        
    finally:
        db.close()

@tool
def analyze_portfolio_trends() -> Dict[str, Any]:
    """
    Analyzes portfolio performance trends across ALL users in the system for regulatory oversight.
    This tool is only available to regulators and provides system-wide portfolio analysis.
    """
    # Check if current user is a regulator
    current_user_id = get_current_user_id_from_context()
    if not current_user_id:
        return {"error": "User not authenticated. Please log in."}
    
    db: Session = SessionLocal()
    try:
        # Verify user is a regulator
        current_user = db.query(models.User).filter(models.User.id == current_user_id).first()
        if not current_user or current_user.role != 'regulator':
            return {"error": "This tool is only available to regulators"}
        
        print(f"🔍 Regulator Tool: Analyzing portfolio trends for regulator {current_user_id}")
        
        # Get all users with pension data
        all_users = db.query(models.PensionData).all()
        
        if not all_users:
            return {"error": "No pension data found in the system"}
        
        # Analyze portfolio trends
        portfolio_types = {}
        return_rates = []
        diversity_scores = []
        
        for user_data in all_users:
            # Portfolio type analysis
            p_type = user_data.pension_type or "Unknown"
            if p_type not in portfolio_types:
                portfolio_types[p_type] = 0
            portfolio_types[p_type] += 1
            
            # Return rate analysis
            if user_data.annual_return_rate:
                return_rates.append(user_data.annual_return_rate)
            
            # Diversity analysis
            if user_data.portfolio_diversity_score:
                diversity_scores.append(user_data.portfolio_diversity_score)
        
        # Calculate statistics
        avg_return_rate = sum(return_rates) / len(return_rates) if return_rates else 0
        avg_diversity = sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0
        
        return {
            "system_analysis": True,
            "total_users": len(all_users),
            "portfolio_types": portfolio_types,
            "performance_metrics": {
                "avg_return_rate": f"{avg_return_rate:.2%}",
                "avg_diversity_score": round(avg_diversity, 3),
                "total_return_rates_analyzed": len(return_rates),
                "total_diversity_scores_analyzed": len(diversity_scores)
            },
            "data_source": "SYSTEM_WIDE_PORTFOLIO_ANALYSIS",
            "note": "This portfolio analysis covers all users in the system for regulatory oversight."
        }
        
    finally:
        db.close()

# Global context variable for user_id (shared across the module)
import contextvars
_user_id_context = contextvars.ContextVar('user_id', default=None)
_query_context = contextvars.ContextVar('query', default=None)

# Fallback global variable for when context doesn't work
_current_user_id = None
_current_query = None

# Helper function to get user_id from context
def get_current_user_id_from_context() -> Optional[int]:
    """
    Get user_id from current request context.
    This should be implemented based on your authentication system.
    """
    try:
        print(f"🔍 Context Debug: Attempting to get user_id from context...")
        
        # Option 1: Request-scoped context (production-ready)
        current_user_id = _user_id_context.get()
        print(f"🔍 Context Debug: Request context value: {current_user_id}")
        
        if current_user_id is not None:
            print(f"🔍 Context: Retrieved user_id={current_user_id} from request context")
            return current_user_id
        
        # Option 2: Global fallback variable
        if _current_user_id is not None:
            print(f"🔍 Context: Retrieved user_id={_current_user_id} from global fallback")
            return _current_user_id
        
        # Option 3: Thread-local storage (fallback for testing)
        import threading
        user_id = getattr(threading.current_thread(), 'user_id', None)
        print(f"🔍 Context Debug: Thread context value: {user_id}")
        
        if user_id is not None:
            print(f"🔍 Context: Retrieved user_id={user_id} from thread context (testing)")
            return user_id
        
        print(f"🔍 Context: No user_id found in context")
        return None
        
    except Exception as e:
        print(f"Error getting user_id from context: {e}")
        return None

def get_current_query_from_context() -> Optional[str]:
    """
    Get the current query from request context.
    """
    try:
        print(f"🔍 Context Debug: Attempting to get query from context...")
        
        # Option 1: Request-scoped context
        current_query = _query_context.get()
        print(f"🔍 Context Debug: Request context query: {current_query}")
        
        if current_query is not None:
            print(f"🔍 Context: Retrieved query='{current_query}' from request context")
            return current_query
        
        # Option 2: Global fallback variable
        if _current_query is not None:
            print(f"🔍 Context: Retrieved query='{current_query}' from global fallback")
            return _current_query
        
        # Option 3: Thread-local storage (fallback for testing)
        import threading
        query = getattr(threading.current_thread(), 'current_query', None)
        print(f"🔍 Context Debug: Thread context query: {query}")
        
        if query is not None:
            print(f"🔍 Context: Retrieved query='{query}' from thread context (testing)")
            return query
        
        print(f"🔍 Context: No query found in context")
        return None
        
    except Exception as e:
        print(f"Error getting query from context: {e}")
        return None

def set_request_user_id(user_id: int):
    """
    Set the current user_id for the current request context.
    This is what your FastAPI endpoint should call.
    """
    try:
        global _current_user_id
        _current_user_id = user_id  # Set global fallback
        _user_id_context.set(user_id)
        print(f"🔍 Context: Set user_id={user_id} in request context and global fallback")
    except Exception as e:
        print(f"Error setting user_id in request context: {e}")
        # Fallback to thread-local for testing
        set_current_user_id(user_id)

def set_request_query(query: str):
    """
    Set the current query for the current request context.
    """
    try:
        global _current_query
        _current_query = query  # Set global fallback
        _query_context.set(query)
        print(f"🔍 Context: Set query='{query}' in request context and global fallback")
    except Exception as e:
        print(f"Error setting query in request context: {e}")
        # Fallback to thread-local for testing
        import threading
        threading.current_thread().current_query = query

def clear_request_user_id():
    """
    Clear the current user_id from the request context.
    This should be called at the end of each request.
    """
    try:
        global _current_user_id, _current_query
        _current_user_id = None  # Clear global fallback
        _current_query = None  # Clear query fallback
        _user_id_context.set(None)
        _query_context.set(None)
        print(f"🔍 Context: Cleared user_id and query from request context and global fallback")
    except Exception as e:
        print(f"Error clearing user_id from request context: {e}")
        # Fallback to thread-local for testing
        clear_current_user_id()

def extract_user_id_from_input(input_value) -> Optional[int]:
    """
    Extract user_id from various input formats and clean them up
    """
    if input_value is None:
        return None
    
    # If it's already an integer, return it
    if isinstance(input_value, int):
        return input_value
    
    # If it's a string, try to extract the number
    if isinstance(input_value, str):
        # Remove common prefixes and extract just the number
        import re
        match = re.search(r'(\d+)', input_value)
        if match:
            user_id = int(match.group(1))
            print(f"🔍 Input Cleanup: Extracted user_id={user_id} from '{input_value}'")
            return user_id
    
    # If it's a dict, try to get user_id
    if isinstance(input_value, dict):
        user_id = input_value.get('user_id')
        if user_id:
            return extract_user_id_from_input(user_id)
    
    print(f"🔍 Input Cleanup: Could not extract user_id from '{input_value}'")
    return None

# Enhanced context management functions from the uploaded version
def set_current_user_id(user_id: int):
    """Set the current user_id for the current thread (for testing)"""
    import threading
    threading.current_thread().user_id = user_id
    print(f"🔍 Context: Set user_id={user_id} in thread context (testing)")

def clear_current_user_id():
    """Clear the current user_id for the current thread"""
    import threading
    if hasattr(threading.current_thread(), 'user_id'):
        delattr(threading.current_thread(), 'user_id')
        print(f"🔍 Context: Cleared user_id from thread context (testing)")

def set_current_query(query: str):
    """Set the current query for the current thread (for testing)"""
    import threading
    threading.current_thread().current_query = query
    print(f"🔍 Context: Set query='{query}' in thread context (testing)")

def clear_current_query():
    """Clear the current query for the current thread"""
    import threading
    if hasattr(threading.current_thread(), 'current_query'):
        delattr(threading.current_thread(), 'current_query')
        print(f"🔍 Context: Cleared query from thread context (testing)")

# Context manager for setting user_id during testing
import threading

def set_current_user_id(user_id: int):
    """Set the current user_id for the current thread (for testing)"""
    threading.current_thread().user_id = user_id

def clear_current_user_id():
    """Clear the current user_id for the current thread"""
    if hasattr(threading.current_thread(), 'user_id'):
        delattr(threading.current_thread(), 'user_id')

def detect_role_based_context(original_query: str, current_user_id: int, db: Session) -> tuple[int, str]:
    """
    Detect the role-based context for a query.
    Returns (target_user_id, context_type) where context_type is:
    - 'self': User is asking about their own data
    - 'client': User is asking about a client's data (for advisors)
    - 'any': User can access any data (for regulators)
    - 'unknown': Unknown context, use current user's data
    """
    print(f"🔍 Role Context Detection: Query='{original_query}', Current User ID={current_user_id}")
    
    # Get current user's role
    current_user = db.query(models.User).filter(models.User.id == current_user_id).first()
    if not current_user:
        print(f"🔍 Role Context: Current user {current_user_id} not found in database")
        return current_user_id, 'unknown'
    
    current_role = current_user.role
    print(f"🔍 Role Context: Current user role is '{current_role}'")
    
    # Extract user ID from query if present
    detected_user_id = None
    if original_query:
        # Look for patterns like "user id X", "user X", "client X", etc.
        import re
        
        # More specific patterns first to avoid false matches
        patterns = [
            r'user\s+id\s+(\d+)',      # "user id 1"
            r'for\s+user\s+id\s+(\d+)', # "for user id 1"
            r'user\s+(\d+)',            # "user 1"
            r'client\s+(\d+)',          # "client 1"
            r'for\s+user\s+(\d+)',      # "for user 1"
            r'(\d+)'                    # fallback: any number
        ]
        
        for pattern in patterns:
            match = re.search(pattern, original_query.lower())
            if match:
                detected_user_id = int(match.group(1))
                print(f"🔍 Role Context: Detected user ID {detected_user_id} from query '{original_query}'")
                break
        
        # Additional validation: if we found a number, make sure it's not the current user's ID
        if detected_user_id == current_user_id:
            print(f"🔍 Role Context: Detected ID {detected_user_id} matches current user - treating as self query")
            detected_user_id = None
    
    # Apply role-based logic
    if current_role == 'resident':
        # Residents can only access their own data
        print(f"🔍 Role Context: Resident role - using own user ID {current_user_id}")
        return current_user_id, 'self'
    
    elif current_role == 'advisor':
        if detected_user_id and detected_user_id != current_user_id:
            # Check if the detected user is a client of this advisor
            client_relationship = db.query(models.AdvisorClient).filter(
                models.AdvisorClient.advisor_id == current_user_id,
                models.AdvisorClient.resident_id == detected_user_id
            ).first()
            
            if client_relationship:
                print(f"🔍 Role Context: Advisor accessing client data (ID: {detected_user_id})")
                return detected_user_id, 'client'
            else:
                print(f"🔍 Role Context: Advisor cannot access user {detected_user_id} - not a client")
                return current_user_id, 'self'
        else:
            # No specific user mentioned or asking about self
            print(f"🔍 Role Context: Advisor using own user ID {current_user_id}")
            return current_user_id, 'self'
    
    elif current_role == 'regulator':
        if detected_user_id:
            # Regulators can access any user's data
            print(f"🔍 Role Context: Regulator accessing user data (ID: {detected_user_id})")
            return detected_user_id, 'client'
        else:
            # No specific user mentioned - use current user (regulator)
            print(f"🔍 Role Context: Regulator using own user ID {current_user_id}")
            return current_user_id, 'self'
    
    else:
        # Unknown role - use current user's data
        print(f"🔍 Role Context: Unknown role '{current_role}' - using current user ID {current_user_id}")
        return current_user_id, 'unknown'

# Export all tools including the new regulator tools
all_pension_tools = [
    analyze_risk_profile,
    detect_fraud,
    project_pension,
    knowledge_base_search,
    analyze_uploaded_document,
    query_knowledge_base,
    # New regulator tools
    analyze_system_wide_risk,
    analyze_system_wide_fraud,
    analyze_geographic_risk,
    analyze_portfolio_trends
]

