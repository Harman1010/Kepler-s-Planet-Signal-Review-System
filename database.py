from sqlalchemy.orm import declarative_base,sessionmaker
from sqlalchemy import create_engine



Base = declarative_base()
eng = create_engine("sqlite:///database.db",connect_args={"check_same_thread" : False})
sessionLocal = sessionmaker(autocommit=False,autoflush=False,bind=eng)

def get_db():
    try:
        db = sessionLocal()
        yield db
    finally:
        db.close()