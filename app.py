from flask import Flask, request, send_file, render_template_string
import pandas as pd
import numpy as np
import joblib
import tempfile
import os
from flask import Flask, request, send_file, render_template

app = Flask(__name__)

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("student_dropout_model.joblib")

# Expected columns
expected_cols = ['Sem1_CGPA','Sem2_CGPA','Sem3_CGPA','Sem4_CGPA','Sem5_CGPA','Sem6_CGPA',
                 'Sem7_CGPA','Sem8_CGPA','Average_CGPA','Total_Reattempts','Annual_Fees',
                 'Fees_Status','Scholarship','Family_Annual_Income_INR','Distance_km','Attendance_Percentage',
                 'ExtraCurricular']

numeric_cols = ['Sem1_CGPA','Sem2_CGPA','Sem3_CGPA','Sem4_CGPA','Sem5_CGPA','Sem6_CGPA',
                'Sem7_CGPA','Sem8_CGPA','Average_CGPA','Total_Reattempts','Annual_Fees',
                'Family_Annual_Income_INR','Distance_km','Attendance_Percentage']

categorical_cols = ['Fees_Status','Scholarship','ExtraCurricular']

# -----------------------------
# Home page
# -----------------------------

@app.route("/")
def index():
    return render_template("index.html")

# -----------------------------
# Prediction route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file:
            return "No file uploaded", 400

        # Save temp
        temp_path = os.path.join(tempfile.gettempdir(), file.filename)
        file.save(temp_path)

        # Read Excel or CSV
        if file.filename.endswith(".csv"):
            df_new = pd.read_csv(temp_path)
        else:
            df_new = pd.read_excel(temp_path)

        # Drop fully empty rows
        df_new = df_new.dropna(how="all").reset_index(drop=True)

        # Add missing columns
        for col in expected_cols:
            if col not in df_new.columns:
                if col in categorical_cols:
                    df_new[col] = 'None'
                else:
                    df_new[col] = 0

        # Fill missing numeric values with training mean if possible
        try:
            numeric_transformer = None
            for name, step in model.named_steps.items():
                if hasattr(step, 'transformers_'):
                    for trans_name, trans, cols in step.transformers_:
                        if trans_name == 'num':
                            numeric_transformer = trans
                            break
            if numeric_transformer is not None and hasattr(numeric_transformer, 'statistics_'):
                means = numeric_transformer.statistics_
                for i, col in enumerate(numeric_cols):
                    if col in df_new.columns:
                        df_new[col] = df_new[col].fillna(means[i])
            else:
                df_new[numeric_cols] = df_new[numeric_cols].fillna(df_new[numeric_cols].mean())
        except:
            df_new[numeric_cols] = df_new[numeric_cols].fillna(df_new[numeric_cols].mean())

        # Fill categorical
        df_new[categorical_cols] = df_new[categorical_cols].fillna('None')

        # -----------------------------
        # Predict
        # -----------------------------
        probs = model.predict_proba(df_new[expected_cols])[:, 1]
        preds = model.predict(df_new[expected_cols])

        # Risk levels
        risk_levels = []
        for p in probs:
            if p <= 0.3:
                risk_levels.append("Low Risk")
            elif p <= 0.7:
                risk_levels.append("Medium Risk")
            else:
                risk_levels.append("High Risk")

        # Reasons
        reasons = []
        for pred, row_tuple in zip(preds, df_new.iterrows()):
            idx, row = row_tuple
            if pred == 1:
                row_reasons = []
                if "Attendance_Percentage" in row and row["Attendance_Percentage"] < 65:
                    row_reasons.append("Low attendance")
                if "Average_CGPA" in row and row["Average_CGPA"] < 6.0:
                    row_reasons.append("Low CGPA")
                if "Total_Reattempts" in row and row["Total_Reattempts"] > 3:
                    row_reasons.append("High reattempts")
                if "Annual_Fees" in row and row["Annual_Fees"] > 150000:
                    row_reasons.append("High fees")
                if "Fees_Status" in row and str(row["Fees_Status"]).lower() != "paid":
                    row_reasons.append("Fees not paid")
                if "Scholarship" in row and str(row["Scholarship"]).strip().lower() == "no":
                    row_reasons.append("No scholarship")
                if "Distance_km" in row and row["Distance_km"] > 50:
                    row_reasons.append("Far from college")
                if "ExtraCurricular" in row:
                    ec = str(row["ExtraCurricular"]).strip().lower()
                    if ec == "none":
                        row_reasons.append("No extracurricular activity")
                    elif ec == "rare":
                        row_reasons.append("Rare participation in extracurriculars")
                if not row_reasons:
                    row_reasons.append("Overall risk factors")
                reasons.append(", ".join(row_reasons))
            else:
                reasons.append("No dropout")

        # Add predictions
        df_new["Predicted_DroppedOut"] = preds
        df_new["Dropout_Probability"] = np.round(probs, 3)
        df_new["Risk_Level"] = risk_levels
        df_new["Predicted_Reasons"] = reasons

        # -----------------------------
        # Save Excel with row coloring
        # -----------------------------
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        with pd.ExcelWriter(temp_file.name, engine='xlsxwriter') as writer:
            df_new.to_excel(writer, index=False, sheet_name="Predictions")
            workbook = writer.book
            worksheet = writer.sheets["Predictions"]

            # Define formats
            format_low = workbook.add_format({'bg_color': '#C6EFCE'})
            format_medium = workbook.add_format({'bg_color': '#FFEB9C'})
            format_high = workbook.add_format({'bg_color': '#FFC7CE'})

            n_cols = len(df_new.columns)
            for row_num, risk in enumerate(risk_levels, start=1):
                fmt = format_low if risk == "Low Risk" else format_medium if risk == "Medium Risk" else format_high
                for col_num in range(n_cols):
                    worksheet.write(row_num, col_num, df_new.iloc[row_num-1, col_num], fmt)

        return send_file(temp_file.name, as_attachment=True, download_name="Predictions.xlsx")

    except Exception as e:
        return str(e), 500

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
