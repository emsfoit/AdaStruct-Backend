from app import db
from sqlalchemy.dialects.postgresql import JSON
import pandas as pd

class DatasetFile(db.Model):
    __tablename__ = 'file'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    filename = db.Column(db.String(250))
    type = db.Column(db.String(10))
    sep = db.Column(db.String(5))
    path = db.Column(db.String(300))
    created_at = db.Column(db.DateTime(128), server_default=db.func.now())
    # Relationship
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'))

    def __repr__(self):
        return '<File filename: {} >'.format(self.filename)

    @staticmethod
    def get_all():
        return DatasetFile.query.all()

    @staticmethod
    def get(id):
        return DatasetFile.query.get(int(id))

    @staticmethod
    def get_header(id):
        file = DatasetFile.get(id)
        columns = list(pd.read_csv(file.path, sep=file.sep).columns)
        return columns

    def to_dict(self):
        header = DatasetFile.get_header(self.id)
        sep = self.sep or ""
        if '\\t' in sep:
            sep = "\t"
        data = {
            'id': self.id,
            'filename': self.filename,
            'project_id': self.project_id,
            'created_at': self.created_at,
            'sep': sep,
            'type': self.type,
            'header': header
        }
        return data