from sqlalchemy import Column, String, Float, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime

Base = declarative_base()

class MLMetrics(Base):
    __tablename__ = 'ml_metrics'
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_name = Column(String)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    training_data_size = Column(Integer)

# Database setup
engine = create_engine('sqlite:///safebrowsing.db')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables
Base.metadata.create_all(bind=engine)
