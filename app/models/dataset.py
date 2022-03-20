from app import db
from sqlalchemy.dialects.postgresql import JSON
import pandas as pd

from app.models.graph import Graph
from app.models.project import Project

class DatasetFile(db.Model):
    __tablename__ = 'file'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    filename = db.Column(db.String(250))
    type = db.Column(db.String(10))
    sep = db.Column(db.String(5))
    path = db.Column(db.String(300))
    created_at = db.Column(db.DateTime(128), server_default=db.func.now())
    # Relationship
    graph_id = db.Column(db.Integer, db.ForeignKey('graph.id'))

    def __repr__(self):
        return '<File filename: {} >'.format(self.filename)

    @classmethod
    def get_user_datasets(cls, current_user):
        return DatasetFile.query.join(Graph).join(Project).filter(Project.company_id == current_user.company_id)

    @staticmethod
    def get_all():
        return DatasetFile.query.all()

    @staticmethod
    def get(id):
        return DatasetFile.query.get(int(id))

    @staticmethod
    def get_header(id):
        file = DatasetFile.get(id)
        try:
            columns = list(pd.read_csv(file.path, sep=file.sep).columns)
            return columns
        except:
            db.session.delete(file)
            db.session.commit()
            return []


    def to_dict(self):
        header = DatasetFile.get_header(self.id)
        sep = self.sep or ""
        if '\\t' in sep:
            sep = "\t"
        data = {
            'id': self.id,
            'filename': self.filename,
            'graph_id': self.graph_id,
            'created_at': self.created_at,
            'sep': sep,
            'type': self.type,
            'header': header
        }
        return data