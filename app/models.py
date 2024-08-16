from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base

db = SQLAlchemy()
Base = declarative_base()

class Bookmark(db.Model):
    __tablename__ = 'bookmarks'

    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String, unique=True, nullable=False)
    title = db.Column(db.String, nullable=False)
    topic = db.Column(db.String, nullable=True)

    def __repr__(self):
        return f'<Bookmark {self.id}: {self.title}>'

def init_db(app):
    db.init_app(app)
    with app.app_context():
        db.create_all()