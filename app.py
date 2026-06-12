import gradio as gr
import pandas as pd
import numpy as np

def create_signal_text(period,duration,depth,snr,prad,teq,steff):

    return f"""
    Orbital Period: {period},
    Transit Duration: {duration},
    Transit Depth: {depth},
    Signal-to-Noise Ratio: {snr},
    Planet Radius: {prad},
    Equilibrium Temperature: {teq},
    Stellar Temperature: {steff}
    """

def analyze_signal(koi_period,koi_time0bk,koi_duration,koi_depth,koi_impact,koi_model_snr,

        koi_prad,koi_teq,koi_insol,koi_steff):

    input_df = pd.DataFrame([{
        "koi_period": koi_period,
        "koi_time0bk": koi_time0bk,
        "koi_duration": koi_duration,
        "koi_depth": koi_depth,
        "koi_impact": koi_impact,
        "koi_model_snr": koi_model_snr,
        "koi_prad": koi_prad,
        "koi_teq": koi_teq,
        "koi_insol": koi_insol,
        "koi_steff": koi_steff
    }])

    pred = best_pipeline.predict(input_df)[0]

    pred_label = le.inverse_transform([pred])[0]

    probabilities = best_pipeline.predict_proba(input_df)[0]

    confirmed_idx = list(le.classes_).index("CONFIRMED")

    confirmed_prob = probabilities[confirmed_idx]

    review_data = engine(confirmed_prob)


    query_text = create_signal_text(
        koi_period,
        koi_duration,
        koi_depth,
        koi_model_snr,
        koi_prad,
        koi_teq,
        koi_steff
    )

    similar_planets = retrieve_similar_planets(
        query_text,
        k=3
    )

    similar_output = ""

    for i, planet in enumerate(similar_planets, 1):

        similar_output += f"""
        {i}. Historical Signal:
        {planet['Planet Name']},

        {planet['Planet Summary']}

        Historical Classification:
        {le.inverse_transform([int(planet['Disposition'])])[0]}

        """

    return (
        pred_label,
        f"{review_data['Confidence Score']}%",
        review_data['Priority'],
        review_data['Review Status'],
        similar_output
    )


interface = gr.Interface(
    fn=analyze_signal,

    inputs=[

        gr.Number(label="Orbital Period"),
        gr.Number(label="Time0bk"),
        gr.Number(label="Transit Duration"),
        gr.Number(label="Transit Depth"),
        gr.Number(label="Impact Parameter"),
        gr.Number(label="Signal-to-Noise Ratio"),
        gr.Number(label="Planet Radius"),
        gr.Number(label="Equilibrium Temperature"),
        gr.Number(label="Insolation"),
        gr.Number(label="Stellar Temperature")

    ],

    outputs=[

        gr.Textbox(label="Predicted Class"),

        gr.Textbox(label="Confidence Score"),

        gr.Textbox(label="Priority Level"),

        gr.Textbox(label="Review Recommendation"),

        gr.Textbox(
            label="Similar Historical Signals",
            lines=20
        ),

    ],

    title="Kepler Planetary Signal Review System",

    description="""
    AI-assisted scientific review prioritization system
    for Kepler exoplanet signal analysis.
    """

)

interface.launch(debug=True)
