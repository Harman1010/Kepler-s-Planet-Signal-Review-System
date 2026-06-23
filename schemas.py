from pydantic import BaseModel, EmailStr


class ValidationRequest(BaseModel):
    koi_period: float
    koi_time0bk: float
    koi_duration: float
    koi_depth: float
    koi_impact: float
    koi_model_snr: float
    koi_prad: float
    koi_teq: float
    koi_insol: float
    koi_steff: float


class HistoryRequest(BaseModel):
    koi_period: float
    koi_depth: float
    koi_prad: float
    koi_steff: float


class Update(BaseModel):
    review_status: str | None = None


class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserLogin(BaseModel):
    email: EmailStr
    password: str