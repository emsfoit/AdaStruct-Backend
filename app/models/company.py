from app import db


class Company(db.Model):
    __tablename__ = 'company'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    name = db.Column(db.String(64), index=True)
    users = db.relationship('User', backref='company', lazy=True)
    projects = db.relationship('Project', backref='company', lazy=True, cascade="all,delete")
    created_at = db.Column(db.DateTime(128), server_default=db.func.now())

    def __init__(self, name: None):
        self.name = name

    def __repr__(self):
        return '<Company Name: {}>'.format(self.name)

    def to_dict(self):
        data = {
            'id': self.id,
            'name': self.name,
            'created_at': self.created_at
        }
        return data

