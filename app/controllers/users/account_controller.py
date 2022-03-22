from flask import Blueprint, request
from app.models.company import Company
from app.models.user import User
from app import db
from app.services.user.decorators import Auth
bp_user_account = Blueprint('/api/user', 'user')



@bp_user_account.route('/sign_up', methods=['POST'])
def create_account():
    data = request.get_json(force=True)
    user = User.query.filter_by(email=data['user_email']).first()
    company = Company.query.filter_by(name=data.get('company_name')).first()

    if user is None and company is None:
        first_name = data.get('first_name')
        last_name = data.get('last_name')
        role = data.get('role')
        email = data.get('user_email')
        password = data.get('password')
        new_user = User(first_name, last_name, role, email, password)
        db.session.add(new_user)
        db.session.commit()

        # Company
        company_name = data.get('company_name')
        new_company = Company(name=company_name)
        db.session.add(new_company)
        db.session.commit()
        new_user.company_id = new_company.id
        db.session.commit()

        return {"message": 'New user is created!!', "data":  new_user.to_dict()}
    return {"message": 'email or company name is not available!!'}, 409


@bp_user_account.route('/login', methods=['POST'])
def login():
    data = request.get_json(force=True)
    email = data.get('user_email')
    password = data.get('password')
    user = User.query.filter_by(email=email).first()
    if user and user.verify_password(user, password):
        auth_token = User.encode_auth_token(user.id)
        if auth_token:
            response_object = {
                'status': 'success',
                'message': 'Successfully logged in.',
                'access_token': auth_token
            }
            return response_object, 200
    else:
        response_object = {
            'status': 'fail',
            'message': 'email or password does not match.'
        }
        return response_object, 401


@bp_user_account.route('/profile', methods=['GET'])
@Auth.verify_token
def get_user(current_user):
    if current_user is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
    return current_user.to_dict()


@bp_user_account.route('remove_account', methods=['DELETE'])
@Auth.verify_token
def delete_user(current_user):
    if current_user is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
    db.session.delete(current_user)
    db.session.commit()
    return {"message": 'user deleted succesfully'}, 200


@bp_user_account.route('/profile',  methods=['PUT'])
@Auth.verify_token
def update_profile(current_user):
    if current_user:
        data = request.get_json(force=True)
        current_user.first_name = data.get('first_name')
        current_user.last_name = data.get('last_name')
        current_user.role = data.get('role')
        current_user.email = data.get('email')
        db.session.commit()
        return {"message": 'profile has been updated!!', "data": current_user.to_dict()}
    response_object = {
        'status': 'fail',
        'message': 'not found.'
    }
    return response_object, 404


