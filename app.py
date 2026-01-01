import joblib
import gradio as gr
import pandas as pd
model = joblib.load("model.joblib")
label_encoder = joblib.load("label_encoder.joblib")
imputerJoblib = joblib.load("imputer.joblib")

CONFIRMED_LABEL = label_encoder.transform(['CONFIRMED'])[0]
THRESHOLD = 0.3

FEATURE_NAMES = [
    'koi_period',
    'koi_time0bk',
    'koi_duration',
    'koi_depth',
    'koi_impact',
    'koi_model_snr',
    'koi_prad',
    'koi_teq',
    'koi_insol',
    'koi_steff'
]


def predict_signal(*inputs):

    X = pd.DataFrame([inputs], columns=FEATURE_NAMES)

    X_imputed = imputerJoblib.transform(X)

    probs = model.predict_proba(X_imputed)[0]
    prob_dict = dict(zip(label_encoder.classes_, probs))

    p_confirmed = prob_dict['CONFIRMED']

    if p_confirmed >= THRESHOLD:
        decision = "🟢 SEND FOR REVIEW"
        explanation = (
            f"P(CONFIRMED) = {p_confirmed:.2f} ≥ {THRESHOLD}. "
            "This signal should be reviewed by scientists."
        )
    else:
        decision = "🔴 SKIP (LOW PRIORITY)"
        explanation = (
            f"P(CONFIRMED) = {p_confirmed:.2f} < {THRESHOLD}. "
            "This signal is low priority for manual review."
        )

    return (
        decision,
        explanation,
        prob_dict
    )
inputs = [
    gr.Number(label=feature) for feature in FEATURE_NAMES
]

outputs = [
    gr.Text(label="Decision"),
    gr.Text(label="Explanation"),
    gr.JSON(label="Class Probabilities")
]

app = gr.Interface(
    fn=predict_signal,
    inputs=inputs,
    outputs=outputs,
    title="Kepler Exoplanet Signal Review System",
    description=(
        "This application uses a trained ML model to prioritize Kepler "
        "telescope signals for human review. A probability threshold of 0.3 "
        "was selected offline to reduce manual workload while retaining most "
        "confirmed planets."
    ),
)

app.launch()
