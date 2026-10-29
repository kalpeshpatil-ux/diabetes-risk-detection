import os
import numpy as np
import pandas as pd

ROOT = "diabetes_risk"
os.makedirs(os.path.join(ROOT, "templates"), exist_ok=True)


def write(path, text):
    with open(os.path.join(ROOT, path), "w", encoding="utf-8") as f:
        f.write(text.lstrip("\n"))
    print("created", os.path.join(ROOT, path))


# ---------------- train_model.py ----------------
write("train_model.py", r'''
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("diabetes.csv")

# In this dataset a 0 in these columns means "missing", so replace with the median
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols] = df[cols].replace(0, np.nan)
df[cols] = df[cols].fillna(df[cols].median())

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred))

joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "features.pkl")
print("Model saved as model.pkl")
''')

# ---------------- app.py ----------------
write("app.py", r'''
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")


def risk_level(p):
    if p < 0.35:
        return "Low", "low"
    if p < 0.65:
        return "Moderate", "mid"
    return "High", "high"


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    values = {}
    error = None
    if request.method == "POST":
        try:
            values = {f: request.form[f] for f in features}
            data = np.array([[float(values[f]) for f in features]])
            prob = model.predict_proba(data)[0][1]
            label, css = risk_level(prob)
            result = {"percent": round(prob * 100, 1), "label": label, "css": css}
        except (ValueError, KeyError):
            error = "Please fill every field with a valid number."
    return render_template("index.html", result=result, values=values, error=error)


if __name__ == "__main__":
    app.run(debug=True)
''')

# ---------------- templates/index.html ----------------
write("templates/index.html", r'''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diabetes Risk Check</title>
<style>
  :root{
    --bg:#f3f7f6; --surface:#ffffff; --ink:#12302e; --muted:#5b7370;
    --line:#d3e0dd; --accent:#0f766e; --low:#2f855a; --mid:#c27c0e; --high:#c0392b;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:Arial,sans-serif;background:var(--bg);color:var(--ink);line-height:1.5}
  .wrap{max-width:960px;margin:0 auto;padding:40px 20px 60px;display:grid;grid-template-columns:1fr 1.15fr;gap:40px}
  h1{font-family:Georgia,serif;font-size:clamp(2rem,5vw,3rem);line-height:1.1;margin:0 0 16px}
  .intro p{color:var(--muted);max-width:38ch}
  .note{font-size:.85rem;border-left:3px solid var(--line);padding-left:12px;margin-top:28px}
  form{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:24px;display:grid;grid-template-columns:1fr 1fr;gap:16px}
  label{display:flex;flex-direction:column;font-size:.85rem;font-weight:bold;gap:4px}
  label small{font-weight:normal;color:var(--muted)}
  input{padding:10px;border:1px solid var(--line);border-radius:8px;font-size:1rem}
  button{grid-column:1/-1;padding:13px;border:0;border-radius:8px;background:var(--accent);color:#fff;font-size:1rem;font-weight:bold;cursor:pointer}
  button:hover{background:#0b5f58}
  .error{grid-column:1/-1;color:var(--high);margin:0}
  .result{grid-column:1/-1;border-radius:10px;padding:18px;border:2px solid;display:flex;align-items:baseline;gap:14px}
  .result strong{font-family:Georgia,serif;font-size:2.4rem}
  .result.low{border-color:var(--low);color:var(--low)}
  .result.mid{border-color:var(--mid);color:var(--mid)}
  .result.high{border-color:var(--high);color:var(--high)}
  @media(max-width:760px){.wrap{grid-template-columns:1fr}form{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <section class="intro">
    <h1>Check your diabetes risk</h1>
    <p>Enter eight health measurements. A machine learning model trained on the Pima Indians Diabetes dataset estimates how likely diabetes is.</p>
    <p class="note">This is a college project, not a medical diagnosis. If you are worried about your health, see a doctor.</p>
  </section>

  <form method="POST">
    {% macro field(name, label, hint, step='any') %}
    <label>{{ label }}<small>{{ hint }}</small>
      <input type="number" step="{{ step }}" min="0" name="{{ name }}" value="{{ values.get(name, '') }}" required>
    </label>
    {% endmacro %}
    {{ field('Pregnancies','Pregnancies','Number of times','1') }}
    {{ field('Glucose','Glucose','mg/dL, fasting') }}
    {{ field('BloodPressure','Blood pressure','Diastolic, mm Hg') }}
    {{ field('SkinThickness','Skin thickness','Triceps fold, mm') }}
    {{ field('Insulin','Insulin','2-hour serum, mu U/ml') }}
    {{ field('BMI','BMI','kg/m²') }}
    {{ field('DiabetesPedigreeFunction','Family history score','Pedigree function, e.g. 0.47') }}
    {{ field('Age','Age','Years','1') }}

    {% if error %}<p class="error" role="alert">{{ error }}</p>{% endif %}
    <button type="submit">Check my risk</button>

    {% if result %}
    <div class="result {{ result.css }}" role="status">
      <strong>{{ result.percent }}%</strong>
      <span>{{ result.label }} risk</span>
    </div>
    {% endif %}
  </form>
</div>
</body>
</html>
''')

# ---------------- requirements.txt ----------------
write("requirements.txt", "flask\npandas\nscikit-learn\njoblib\n")

# ---------------- diabetes.csv (sample data if you don't have the real one) ----------------
csv_path = os.path.join(ROOT, "diabetes.csv")
if not os.path.exists(csv_path):
    rng = np.random.default_rng(42)
    n = 768
    df = pd.DataFrame({
        "Pregnancies": rng.integers(0, 14, n),
        "Glucose": rng.normal(120, 30, n).clip(50, 200).round(),
        "BloodPressure": rng.normal(72, 12, n).clip(40, 120).round(),
        "SkinThickness": rng.normal(29, 10, n).clip(7, 70).round(),
        "Insulin": rng.normal(120, 70, n).clip(15, 600).round(),
        "BMI": rng.normal(32, 6.5, n).clip(18, 60).round(1),
        "DiabetesPedigreeFunction": rng.gamma(2, 0.25, n).clip(0.08, 2.4).round(3),
        "Age": rng.integers(21, 80, n),
    })
    score = (0.035 * (df.Glucose - 120) + 0.08 * (df.BMI - 32)
             + 0.03 * (df.Age - 33) + 1.2 * (df.DiabetesPedigreeFunction - 0.47) - 0.6)
    df["Outcome"] = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)
    df.to_csv(csv_path, index=False)
    print("created", csv_path, "(SAMPLE data - replace with the real Kaggle diabetes.csv)")

print("\nDone! Next steps:")
print("  cd diabetes_risk")
print("  pip install -r requirements.txt")
print("  python train_model.py")
print("  python app.py   ->  open http://127.0.0.1:5000") 
