from app import db
from sqlalchemy.dialects.postgresql import JSON

class Project(db.Model):
    __tablename__ = 'project'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    name = db.Column(db.String(128), index=True, nullable=False)
    graph_setting = db.Column(JSON)
    created_at = db.Column(db.DateTime(128), server_default=db.func.now())
    # Relationship
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'))
    files = db.relationship('DatasetFile', backref='project', lazy=True, cascade="all,delete" )

    def __repr__(self):
        return '<Project Name: {} >'.format(self.name)

    @staticmethod
    def get_all():
        return Project.query.all()

    @staticmethod
    def get(id):
        return Project.query.get(int(id))

    def to_dict(self):
        data = {
            'id': self.id,
            'name': self.name,
            'graph_setting': self.graph_setting,
            'created_at': self.created_at
        }
        return data

    def get_name(self):
        data = {
            'id': self.id,
            'name': self.name,
        }
        return data