from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app import db, login_manager
from flask_jwt_extended import create_access_token
from datetime import datetime, timedelta
import jwt
from app import Config
key = Config.SECRET_KEY


class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    company_id = db.Column(db.Integer, db.ForeignKey('company.id'))
    first_name = db.Column(db.String(64), index=True)
    last_name = db.Column(db.String(64), index=True)
    role = db.Column(db.String(64), index=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime(128), server_default=db.func.now())

    def __init__(self, first_name: None, last_name: None, role: None, email: None, password):
        self.first_name = first_name
        self.last_name = last_name
        self.email = email
        self.password_hash = generate_password_hash(password)
        self.role = role

    @staticmethod
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def __repr__(self):
        return '<User Email: {}, Name: {}, Role {}, Password Hash {}>'.format(self.email, self.first_name + self.last_name, self.role, self.password_hash)

    @login_manager.user_loader
    def get(id):
        return User.query.get(int(id))

    @staticmethod
    def get_all():
        return User.query.all()

    @staticmethod
    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        data = {
            'id': self.id,
            'comany': self.company_id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'email': self.email,
            'role': self.role,
            'created_at': self.created_at
        }
        return data

    @staticmethod
    def encode_auth_token(id):
        """
        Generates the Auth Token
        :return: string
        """
        try:
            payload = {
                'exp': datetime.utcnow() + timedelta(days=1, seconds=5),
                'iat': datetime.utcnow(),
                'user_id': id
            }
            return jwt.encode(
                payload,
                key,
                algorithm='HS256'
            )
        except Exception as e:
            return e

    @staticmethod
    def decode_auth_token(auth_token):
        """
        Decodes the auth token
        :param auth_token:
        :return: integer|string
        """
        try:
            payload = jwt.decode(auth_token, key, algorithms=['HS256'])
            return payload['user_id']
        except jwt.ExpiredSignatureError:
            return 'Signature expired. Please log in again.'
        except jwt.InvalidTokenError:
            return 'Invalid token. Please log in again.'