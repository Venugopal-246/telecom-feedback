from sqlmodel import SQLModel, create_engine


DATABASE_URL = "sqlite:///./voicers.db"
engine = create_engine(DATABASE_URL, echo=False)


def init_db():
	from .models import Feedback
	SQLModel.metadata.create_all(engine)