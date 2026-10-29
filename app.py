"""
=============================================================================
Project Title : Diabetes Risk Detection System
Subject       : Computing in Python (College Project)
File          : app.py
Description   : Flask Web Application that serves:
                - /            : Main Diabetes Risk Prediction Form & Results
                - /performance : Model Comparison, Accuracy & Feature Importance
                - /about       : Project Documentation, Dataset & Viva Questions
                - /api/predict : JSON endpoint for instant prediction calculation
=============================================================================
"""

import os
import json
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
from ml_models import NativeLogisticRegression, NativeDecisionTree, NativeRandomForest

app = Flask(__name__)

# Feature names expected by the model
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

# Field metadata with clinical ranges and user-friendly labels
FIELD_INFO = {
    "Pregnancies": {
        "label": "Pregnancies",
        "hint": "Number of times pregnant (0 for males or nulliparous females)",
        "unit": "Count",
        "min": 0, "max": 20, "step": 1, "default": 1,
        "normal_range": "0 - 4"
    },
    "Glucose": {
        "label": "Fasting Glucose",
        "hint": "Plasma glucose concentration after fasting",
        "unit": "mg/dL",
        "min": 40, "max": 300, "step": 1, "default": 110,
        "normal_range": "70 - 99 mg/dL (Normal), 100 - 125 (Pre-diabetes)"
    },
    "BloodPressure": {
        "label": "Diastolic Blood Pressure",
        "hint": "Diastolic pressure (resting pressure between heartbeats)",
        "unit": "mm Hg",
        "min": 30, "max": 180, "step": 1, "default": 72,
        "normal_range": "< 80 mm Hg (Normal), 80 - 89 (Stage 1)"
    },
    "SkinThickness": {
        "label": "Triceps Skinfold Thickness",
        "hint": "Thickness of skinfold at triceps muscle to estimate body fat",
        "unit": "mm",
        "min": 5, "max": 99, "step": 1, "default": 23,
        "normal_range": "15 - 30 mm (Average)"
    },
    "Insulin": {
        "label": "2-Hour Serum Insulin",
        "hint": "Serum insulin measured 2 hours after glucose challenge",
        "unit": "μU/mL",
        "min": 10, "max": 900, "step": 1, "default": 85,
        "normal_range": "16 - 166 μU/mL"
    },
    "BMI": {
        "label": "Body Mass Index (BMI)",
        "hint": "Weight in kg divided by height in meters squared",
        "unit": "kg/m²",
        "min": 10.0, "max": 70.0, "step": 0.1, "default": 27.5,
        "normal_range": "18.5 - 24.9 (Normal), 25 - 29.9 (Overweight), ≥ 30 (Obese)"
    },
    "DiabetesPedigreeFunction": {
        "label": "Diabetes Pedigree Function",
        "hint": "Genetic risk score based on family history of diabetes",
        "unit": "Score (0 - 2.5)",
        "min": 0.05, "max": 2.50, "step": 0.01, "default": 0.45,
        "normal_range": "0.08 - 0.50 (Typical), > 0.50 (Elevated Family History)"
    },
    "Age": {
        "label": "Age",
        "hint": "Age of the individual in completed years",
        "unit": "Years",
        "min": 15, "max": 120, "step": 1, "default": 35,
        "normal_range": "Risk increases with age (≥ 45 has higher risk)"
    }
}


def load_artifacts():
    """Safely loads model, features, and training metadata."""
    model = None
    features = FEATURE_NAMES
    model_info = {}

    if os.path.exists("model.pkl"):
        try:
            model = joblib.load("model.pkl")
        except Exception as e:
            print(f"[ERROR] Could not load model.pkl: {e}")

    if os.path.exists("features.pkl"):
        try:
            features = joblib.load("features.pkl")
        except Exception as e:
            print(f"[WARN] Could not load features.pkl: {e}")

    if os.path.exists("model_info.json"):
        try:
            with open("model_info.json", "r", encoding="utf-8") as f:
                model_info = json.load(f)
        except Exception as e:
            print(f"[WARN] Could not load model_info.json: {e}")

    return model, features, model_info


def calculate_risk(model, features, values_dict):
    """
    Takes a dictionary of feature values, passes them through the trained model,
    and returns probability, risk level, css category, and tailored health tips.
    """
    # Create numpy array in exact order expected by the model
    input_vector = []
    for f in features:
        val = float(values_dict[f])
        input_vector.append(val)

    data = np.array([input_vector])

    # Predict probability of diabetes (class 1)
    if model is not None and hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(data)[0][1])
    elif model is not None:
        pred = model.predict(data)[0]
        prob = 0.85 if pred == 1 else 0.15
    else:
        # High-precision calibrated Pima Indians logistical model formula
        # logit(p) = -8.404 + 0.123*Preg + 0.035*Gluc - 0.013*BP + 0.0006*Skin - 0.001*Ins + 0.089*BMI + 0.945*Ped + 0.014*Age
        z = (
            -8.404
            + 0.123 * float(values_dict.get("Pregnancies", 0))
            + 0.035 * float(values_dict.get("Glucose", 120))
            - 0.013 * float(values_dict.get("BloodPressure", 70))
            + 0.0006 * float(values_dict.get("SkinThickness", 20))
            - 0.001 * float(values_dict.get("Insulin", 80))
            + 0.089 * float(values_dict.get("BMI", 25))
            + 0.945 * float(values_dict.get("DiabetesPedigreeFunction", 0.47))
            + 0.014 * float(values_dict.get("Age", 33))
        )
        prob = float(1.0 / (1.0 + np.exp(-z)))

    # Determine risk category
    percent = round(prob * 100, 1)
    if prob < 0.35:
        risk_label = "Low Risk"
        css_class = "low"
        badge_color = "#10b981"  # Emerald Green
        summary = "Your measurements suggest a low statistical risk of diabetes."
    elif prob < 0.65:
        risk_label = "Moderate Risk"
        css_class = "mid"
        badge_color = "#f59e0b"  # Amber Orange
        summary = "Your measurements indicate borderline / pre-diabetic risk markers."
    else:
        risk_label = "High Risk"
        css_class = "high"
        badge_color = "#ef4444"  # Crimson Red
        summary = "Your measurements strongly correlate with high risk markers for diabetes."

    # Generate 3-4 actionable lifestyle and health tips
    tips = generate_health_tips(risk_label, values_dict)

    # Metric breakdown showing status of each input
    breakdown = []
    for f in features:
        val = float(values_dict[f])
        info = FIELD_INFO[f]
        status = "normal"
        status_label = "Normal"

        if f == "Glucose":
            if val >= 126:
                status, status_label = "danger", "High (≥126)"
            elif val >= 100:
                status, status_label = "warning", "Borderline (100-125)"
        elif f == "BMI":
            if val >= 30:
                status, status_label = "danger", "Obese (≥30)"
            elif val >= 25:
                status, status_label = "warning", "Overweight (25-29.9)"
        elif f == "BloodPressure":
            if val >= 90:
                status, status_label = "danger", "High (≥90)"
            elif val >= 80:
                status, status_label = "warning", "Elevated (80-89)"

        breakdown.append({
            "feature": f,
            "label": info["label"],
            "value": val,
            "unit": info["unit"],
            "status": status,
            "status_label": status_label
        })

    return {
        "percent": percent,
        "label": risk_label,
        "css": css_class,
        "color": badge_color,
        "summary": summary,
        "tips": tips,
        "breakdown": breakdown
    }


def generate_health_tips(risk_level, values):
    """Generates 3 to 4 clinical and lifestyle guidance tips."""
    glucose = float(values.get("Glucose", 100))
    bmi = float(values.get("BMI", 24))
    bp = float(values.get("BloodPressure", 75))

    tips = []
    if risk_level == "Low Risk":
        tips.append("Maintain an active routine: Aim for at least 150 minutes of moderate aerobic exercise (e.g., brisk walking or swimming) weekly.")
        tips.append("Nutrient-rich diet: Focus on whole grains, legumes, leafy greens, and lean proteins while minimizing refined sugars.")
        tips.append("Routine monitoring: Continue regular annual preventative health checkups and monitor fasting blood sugar periodically.")
        tips.append("Restorative sleep: Ensure 7 to 8 hours of quality sleep each night to help regulate metabolic hormones.")
    elif risk_level == "Moderate Risk":
        tips.append("Consult your doctor: Schedule a comprehensive fasting plasma glucose (FPG) or HbA1c screening.")
        if bmi >= 25:
            tips.append(f"Weight management: With a BMI of {bmi:.1f}, losing just 5% to 7% of body weight can reduce the risk of diabetes by up to 58%.")
        else:
            tips.append("Build muscle mass: Combine aerobic workouts with resistance training 2 days a week to boost cellular insulin sensitivity.")
        if glucose >= 100:
            tips.append(f"Carbohydrate control: Your glucose ({glucose:.0f} mg/dL) is in the pre-diabetes bracket. Replace sugary drinks and processed flour with high-fiber meals.")
        else:
            tips.append("Portion control: Manage carbohydrate portions throughout the day to avoid blood sugar spikes.")
        if bp >= 80:
            tips.append(f"Blood pressure vigilance: Your diastolic BP is {bp:.0f} mm Hg. Cut dietary sodium intake and incorporate stress reduction practices.")
    else:  # High Risk
        tips.append("Medical attention recommended: Schedule an evaluation with a physician or endocrinologist for clinical diagnosis.")
        tips.append("Targeted medical nutrition therapy: Consult a registered dietitian to establish a low-glycemic meal plan and eliminate simple sugars.")
        tips.append("Daily physical activity: Perform 30 minutes of light-to-moderate exercise (such as a brisk walk) daily, ideally after meals.")
        tips.append("Comprehensive cardiovascular monitoring: Actively monitor blood pressure, cholesterol levels, and blood sugar in collaboration with a healthcare provider.")

    return tips[:4]


# =============================================================================
# Routes
# =============================================================================

@app.route("/", methods=["GET", "POST"])
def index():
    """Main route: renders the prediction form and handles form submissions."""
    model, features, model_info = load_artifacts()

    result = None
    values = {}
    error = None

    # Default preset demo values for ease of testing
    demo_profiles = {
        "healthy": {
            "Pregnancies": 1, "Glucose": 88, "BloodPressure": 68,
            "SkinThickness": 20, "Insulin": 60, "BMI": 22.4,
            "DiabetesPedigreeFunction": 0.25, "Age": 28
        },
        "moderate": {
            "Pregnancies": 3, "Glucose": 118, "BloodPressure": 82,
            "SkinThickness": 28, "Insulin": 130, "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.48, "Age": 45
        },
        "high": {
            "Pregnancies": 6, "Glucose": 168, "BloodPressure": 88,
            "SkinThickness": 38, "Insulin": 220, "BMI": 35.4,
            "DiabetesPedigreeFunction": 0.85, "Age": 52
        }
    }

    if request.method == "POST":
        try:
            # Extract and validate form inputs
            for f in features:
                raw_val = request.form.get(f, "").strip()
                if raw_val == "":
                    raise ValueError(f"Field '{FIELD_INFO[f]['label']}' cannot be empty.")
                val = float(raw_val)
                if val < 0:
                    raise ValueError(f"Field '{FIELD_INFO[f]['label']}' cannot be negative.")
                values[f] = val

            # Perform prediction
            result = calculate_risk(model, features, values)
            if model is None:
                result["summary"] += " (Computed using calibrated Pima model. Run 'python train_model.py' to use trained Random Forest .pkl)"

        except ValueError as ve:
            error = f"Validation Error: {str(ve)}"
        except Exception as ex:
            error = f"An unexpected error occurred: {str(ex)}"


    return render_template(
        "index.html",
        result=result,
        values=values,
        error=error,
        field_info=FIELD_INFO,
        features=features,
        demo_profiles=demo_profiles,
        model_ready=(model is not None)
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """AJAX JSON endpoint for instantaneous risk calculation without page reload."""
    model, features, _ = load_artifacts()
    if model is None:
        return jsonify({"success": False, "error": "Model not trained yet. Run train_model.py first."}), 400

    try:
        data = request.get_json(force=True)
        values = {}
        for f in features:
            if f not in data:
                return jsonify({"success": False, "error": f"Missing feature: {f}"}), 400
            val = float(data[f])
            if val < 0:
                return jsonify({"success": False, "error": f"Value for {f} cannot be negative"}), 400
            values[f] = val

        result = calculate_risk(model, features, values)
        return jsonify({"success": True, "result": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


@app.route("/performance")
def performance():
    """Renders the model comparison, accuracy, and feature importance page."""
    _, _, model_info = load_artifacts()
    return render_template("performance.html", model_info=model_info)


@app.route("/about")
def about():
    """Renders the project documentation, dataset explanation, and viva Q&A guide."""
    return render_template("about.html", field_info=FIELD_INFO)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*60}")
    print(" Diabetes Risk Detection System - Web Server Starting")
    print(f" Live on: http://127.0.0.1:{port}")
    print(f"{'='*60}\n")
    app.run(host="0.0.0.0", port=port, debug=False)

