from app import db
from sqlalchemy.dialects.postgresql import JSON
from .project import Project

class Graph(db.Model):
    __tablename__ = 'graph'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    name = db.Column(db.String(128), index=True, nullable=False)
    settings = db.Column(JSON)
    path = db.Column(db.String(300))
    created_at = db.Column(db.DateTime(128), server_default=db.func.now())
    # Relationship
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'))
    files = db.relationship('DatasetFile', backref='graph', lazy=True, cascade="all,delete" )

    def __repr__(self):
        return '<Graph Name: {} >'.format(self.name)

    @classmethod
    def get_users_graph(cls, current_user):
        return Graph.query.join(Project).filter(Project.company_id == current_user.company_id)
        
    @staticmethod
    def get_all():
        return Graph.query.all()

    @staticmethod
    def get(id):
        return Graph.query.get_or_404(int(id))

    def to_dict(self):
        data = {
            'id': self.id,
            'name': self.name,
            'settings': self.settings,
            'path': self.path,
            'created_at': self.created_at
        }
        return data
