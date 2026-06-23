from fastapi import (
    APIRouter,
    Depends,
    HTTPException
)

from sqlalchemy.orm import Session

from database import get_db

from models import User

from schemas import UserCreate

from auth import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user
)

from fastapi.security import (
    OAuth2PasswordRequestForm
)

router = APIRouter()


@router.post("/register")
def register(
    data: UserCreate,
    db: Session = Depends(get_db)
):

    existing_user = db.query(User).filter(
        User.email == data.email
    ).first()

    if existing_user:

        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    new_user = User(
        email=data.email,
        hashed_password=hash_password(
            data.password
        )
    )

    db.add(new_user)
    db.commit()

    return {
        "message":
        "User registered successfully"
    }


@router.post("/login")
def login(
    form_data:
    OAuth2PasswordRequestForm = Depends(),

    db: Session = Depends(get_db)
):

    user = db.query(User).filter(
        User.email ==
        form_data.username
    ).first()

    if not user:

        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    if not verify_password(
        form_data.password,
        user.hashed_password
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    token = create_access_token(
        {"sub": str(user.id)}
    )

    return {
        "access_token": token,
        "token_type": "bearer"
    }


@router.get("/me")
def me(
    current_user: User =
    Depends(get_current_user)
):

    return {

        "id":
        current_user.id,

        "email":
        current_user.email

    }