from sqlalchemy import create_engine,Column, Integer, String,Float,DateTime
from sqlalchemy.orm import declarative_base,sessionmaker
from datetime import datetime


DATABASE_URL="sqlite:////app/data/churn.db"

engine=create_engine(DATABASE_URL,connect_args={"check_same_thread": False})

SessionLocal=sessionmaker(bind=engine)

Base=declarative_base()

class Prediction(Base):
  __tablename__="predictions"
  id=Column(Integer,primary_key=True,index=True)
  client_id=Column(String,index=True,nullable=True)
  churn_proba=Column(Float)
  label=Column(String)
  monthly_charges=Column(Float,nullable=True)
  create_at=Column(DateTime,default=datetime.utcnow)

def init_db():
  Base.metadata.create_all(bind=engine)

def get_db():
  db=SessionLocal()
  try:
    yield db
  finally:
    db.close()
                     