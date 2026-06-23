from fastapi import APIRouter

from schemas import HistoryRequest

from services.retrieval import (
    create_input_signal,
    retrieve_similar_planets
)

router = APIRouter()


@router.post("/history")
def history(data: HistoryRequest):

    query_text = create_input_signal(
        data.koi_period,
        data.koi_depth,
        data.koi_prad,
        data.koi_steff
    )

    similar_planets = retrieve_similar_planets(
        query_text,
        k=3
    )

    return {

        "Historical Similar Planets":
            similar_planets

    }