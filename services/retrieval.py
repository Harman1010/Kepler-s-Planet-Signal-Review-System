import joblib
import pandas as pd
import numpy as np
import faiss

from sentence_transformers import (
    SentenceTransformer
)

model_embed = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

faiss_index = faiss.read_index(
    "model_files/faiss_index.bin"
)

planet_texts = joblib.load(
    "model_files/planet_texts.joblib"
)

df = pd.read_csv(
    "model_files/history_df.csv"
)

le = joblib.load(
    "model_files/label_encoder.joblib"
)


def create_input_signal(
    period,
    depth,
    prad,
    steff
):

    return f"""
    Orbital Period: {period},
    Transit Depth: {depth},
    Planet Radius: {prad},
    Stellar Temperature: {steff}
    """


def retrieve_similar_planets(
    input_text,
    k=3
):

    query_embedding = model_embed.encode(
        [input_text]
    )

    distances, indices = (
        faiss_index.search(
            np.array(query_embedding)
            .astype("float32"),
            k
        )
    )

    results = []

    for idx in indices[0]:

        idx = int(idx)

        results.append({

            "Planet Name":
                str(df.iloc[idx]["kepoi_name"]),

            "Planet Summary":
                str(planet_texts[idx]),

            "Disposition":
                le.inverse_transform(
                    [int(
                        df.iloc[idx]
                        ["koi_disposition"]
                    )]
                )[0]
        })

    return results