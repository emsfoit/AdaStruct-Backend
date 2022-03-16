from flask import request
from app.models.user import User
from functools import wraps


class Auth():
    @staticmethod
    def get_logged_in_user():
        auth_token = request.headers.get('Authorization')
        if auth_token:
            resp = User.decode_auth_token(auth_token)
            if not isinstance(resp, str):
                user = User.query.filter_by(id=resp).first()
                if user:
                    response_object = {
                        'status': 'success',
                        'data': {
                            'user': user,
                            'user_id': user.id,
                            'email': user.email
                        }
                    }
                    return response_object, 200
            response_object = {
                'status': 'fail',
                'message': resp
            }
            return response_object, 401
        else:
            response_object = {
                'status': 'fail',
                'message': 'Provide a valid auth token.'
            }
            return response_object, 401

    def verify_token(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            data, status = Auth.get_logged_in_user()
            token = data.get('data')
            if not token:
                return data, status
            current_user = token.get('user')
            return f(current_user, *args, **kwargs)
        return decorated