from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, PickleType

db = SQLAlchemy()
Base = declarative_base()

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()

class Bookmark(Base):
    __tablename__ = 'bookmarks'

    id = Column(Integer, primary_key=True)
    url = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    embedding = Column(PickleType, nullable=False)
    topic = Column(String, nullable=False)

    # Add this line to associate the model with SQLAlchemy
    query = db.session.query_property()