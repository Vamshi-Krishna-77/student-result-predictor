from django.shortcuts import render
import os
import joblib
import pandas as pd

# Load model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'predictor', 'models', 'xgb_exam_model.pkl')
model = joblib.load(MODEL_PATH)


def home(request):
    result = None
    confidence = None
    feature_data = None

    if request.method == "POST":
        try:
            data = {
                "prev_gpa": float(request.POST.get("prev_gpa")),
                "prereq_score": float(request.POST.get("prereq_score")),
                "midterm_score": float(request.POST.get("midterm_score")),
                "quiz_avg": float(request.POST.get("quiz_avg")),
                "attend_rate": float(request.POST.get("attend_rate")),
                "lab_attend": float(request.POST.get("lab_attend")),
                "study_hrs": float(request.POST.get("study_hrs")),
                "entrance_exam": float(request.POST.get("entrance_exam")),
                "is_fulltime": int(request.POST.get("is_fulltime")),
                "year_of_study": int(request.POST.get("year_of_study")),
            }

            df = pd.DataFrame([data])

            # 🔥 Feature Engineering
            df["performance_avg"] = (df["prereq_score"] + df["midterm_score"] + df["quiz_avg"]) / 3
            df["attendance_avg"] = (df["attend_rate"] + df["lab_attend"]) / 2
            df["study_index"] = df["study_hrs"] * (df["attendance_avg"] / 100)
            df["efficiency_score"] = df["performance_avg"] * (df["study_index"] / 10)

            # ✅ Prediction
            prediction = model.predict(df)[0]
            prob = model.predict_proba(df)[0][prediction]

            confidence = round(prob * 100, 2)
            result = "PASS" if prediction == 1 else "FAIL"

            # ✅ Feature Importance (Safe)
            try:
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "named_steps"):
                    clf = model.named_steps.get("clf")
                    importances = clf.feature_importances_
                else:
                    importances = [0] * len(df.columns)

                feature_names = df.columns.tolist()
                feature_data = sorted(
                    zip(feature_names, importances),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

            except:
                feature_data = None

        except Exception as e:
            result = f"Error: {str(e)}"

    return render(request, 'index.html', {
        "result": result,
        "confidence": confidence,
        "features": feature_data
    })