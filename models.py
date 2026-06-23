from sqlalchemy import Integer, Float, String,ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


class Model(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    confidence: Mapped[float] = mapped_column(Float)

    prediction: Mapped[str] = mapped_column(String)

    priority: Mapped[str] = mapped_column(String)

    review_status: Mapped[str] = mapped_column(String)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id")
    )

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True
    )

    email: Mapped[str] = mapped_column(
        String,
        unique=True,
        index=True
    )

    hashed_password: Mapped[str] = mapped_column(
        String
    )