import joblib
import pandas as pd

model = joblib.load("model_files/finalModel.joblib")
le = joblib.load("model_files/label_encoder.joblib")


def engine(prob):
    confidence = round(prob * 100, 2)

    if confidence >= 70:
        priority = "Critical"
        review_status = "Immediate Review"

    elif confidence >= 50:
        priority = "High"
        review_status = "Scientist Validation"

    elif confidence >= 30:
        priority = "Medium"
        review_status = "Review Queue"

    else:
        priority = "Low"
        review_status = "Auto-Filtered"

    return {
        "Confidence Score": confidence,
        "Priority": priority,
        "Review Status": review_status
    }


def predict_signal(data):

    input_df = pd.DataFrame([{
        "koi_period": data.koi_period,
        "koi_time0bk": data.koi_time0bk,
        "koi_duration": data.koi_duration,
        "koi_depth": data.koi_depth,
        "koi_impact": data.koi_impact,
        "koi_model_snr": data.koi_model_snr,
        "koi_prad": data.koi_prad,
        "koi_teq": data.koi_teq,
        "koi_insol": data.koi_insol,
        "koi_steff": data.koi_steff
    }])

    prediction = model.predict(input_df)[0]

    prediction_label = le.inverse_transform(
        [prediction]
    )[0]

    probabilities = model.predict_proba(
        input_df
    )[0]

    confirmed_idx = list(
        le.classes_
    ).index("CONFIRMED")

    confirmed_prob = float(
        probabilities[confirmed_idx]
    )

    review_data = engine(
        confirmed_prob
    )

    return prediction_label, review_data