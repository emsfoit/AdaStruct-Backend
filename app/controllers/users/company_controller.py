from flask import Blueprint, request
from app.models.company import Company
from app import db
from app.services.user.decorators import Auth
bp_user_companies = Blueprint('companies', 'company')


# !This file can be used by Admin if needed

@bp_user_companies.route('/companies', methods=['POST'])
@Auth.verify_token
def create_company(current_user):
    data = request.get_json(force=True).get('company')
    name = data.get('name')
    new_company = Company(name=name)
    db.session.add(new_company)
    db.session.commit()
    return {"message": 'New company has been created!!', "data": new_company.to_dict()}


@bp_user_companies.route('/companies', methods=['GET'])  # return all companies
@Auth.verify_token
def get_all_companies(current_user):
    companies = Company.query.all()
    result = [company.to_dict() for company in companies]
    return {'data': result}, 200


@bp_user_companies.route('companies/<int:id>', methods=['GET'])
@Auth.verify_token
def get_company(current_user, id):
    company = company.get(id)
    if company is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
    return company.to_dict()


@bp_user_companies.route('companies/<int:id>', methods=['DELETE'])
@Auth.verify_token
def delete_company(current_user, id):
    company = Company.get(id)
    if company is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
    db.session.delete(company)
    db.session.commit()
    return {"message": 'company deleted successfully'}, 200
