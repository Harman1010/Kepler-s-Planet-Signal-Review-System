from fastapi import (
    APIRouter,
    Depends,
    HTTPException
)

from sqlalchemy.orm import Session

from database import get_db

from models import (
    Model,
    User
)

from schemas import (
    ValidationRequest,
    Update
)

from auth import (
    get_current_user
)

from services.predictor import (
    predict_signal
)

router = APIRouter()


@router.get("/predictions")
def get_predictions(

    db: Session =
    Depends(get_db),

    current_user: User =
    Depends(get_current_user)

):

    predictions = (

        db.query(Model)

        .filter(
            Model.user_id ==
            current_user.id
        )

        .all()

    )

    return predictions


@router.post("/predict")
def predict(

    data: ValidationRequest,

    db: Session =
    Depends(get_db),

    current_user: User =
    Depends(get_current_user)

):

    prediction_label, review_data = (
        predict_signal(data)
    )

    record = Model(

        confidence=
        review_data[
            "Confidence Score"
        ],

        prediction=
        prediction_label,

        priority=
        review_data[
            "Priority"
        ],

        review_status=
        review_data[
            "Review Status"
        ],

        user_id=
        current_user.id

    )

    db.add(record)

    db.commit()

    db.refresh(record)

    return {

        "predicted_class":
        prediction_label,

        "Confidence Score":
        review_data[
            "Confidence Score"
        ],

        "Priority":
        review_data[
            "Priority"
        ],

        "Review Recommendation":
        review_data[
            "Review Status"
        ]

    }


@router.patch(
    "/predictions/{prediction_id}"
)
def patch(

    prediction_id: int,

    data: Update,

    db: Session =
    Depends(get_db),

    current_user: User =
    Depends(get_current_user)

):

    stmt = (

        db.query(Model)

        .filter(

            Model.id ==
            prediction_id,

            Model.user_id ==
            current_user.id

        )

        .first()

    )

    if stmt is None:

        raise HTTPException(

            status_code=404,

            detail=
            "Prediction not found"

        )

    stmt.review_status = (
        data.review_status
    )

    db.commit()

    db.refresh(stmt)

    return stmt


@router.delete(
    "/predictions/{prediction_id}"
)
def delete_record(

    prediction_id: int,

    db: Session =
    Depends(get_db),

    current_user: User =
    Depends(get_current_user)

):

    stmt = (

        db.query(Model)

        .filter(

            Model.id ==
            prediction_id,

            Model.user_id ==
            current_user.id

        )

        .first()

    )

    if stmt is None:

        raise HTTPException(

            status_code=404,

            detail=
            "Record not found"

        )

    db.delete(stmt)

    db.commit()

    return {

        "Message":
        "Record Deleted",

        "ID":
        prediction_id

    }