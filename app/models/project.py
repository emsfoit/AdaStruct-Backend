from app import db
from sqlalchemy.dialects.postgresql import JSON

class Project(db.Model):
    __tablename__ = 'project'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    name = db.Column(db.String(128), index=True, nullable=False)
    created_at = db.Column(db.DateTime(128), server_default=db.func.now())
    # Relationship
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'))
    graphs = db.relationship('Graph', backref='project', lazy=True, cascade="all,delete" )

    def __repr__(self):
        return '<Project Name: {} >'.format(self.name)

    @classmethod
    def get_user_projects(cls, current_user):
        return Project.query.filter(Project.company_id == current_user.company_id)

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
            'created_at': self.created_at
        }
        return data
