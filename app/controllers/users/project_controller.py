from flask import Blueprint, request
from app.models.project import Project
from app import db
from app.services.user.decorators import Auth

bp_user_projects = Blueprint('projects', 'project')

@bp_user_projects.route('/projects', methods=['POST'])
@Auth.verify_token
def create_project(current_user):
    data = request.get_json(force=True)
    name = data.get('name')
    company_id = current_user.company_id
    project = Project.get_user_projects(current_user).filter(Project.name == name).first()
    if not project:
        new_project = Project(name=name, company_id=company_id)
        db.session.add(new_project)
        db.session.commit()
        return {"message": 'New project has been created!!', "data": new_project.to_dict()}
    response_object = {
        "name": ["name is invalid"]
    }
    return response_object, 400
    

@bp_user_projects.route('/projects', methods=['GET'])  # return all projects
@Auth.verify_token
def get_all_projects(current_user):
    projects = Project.get_user_projects(current_user).all()
    result = [project.to_dict() for project in projects]
    return {'data': result}, 200


@bp_user_projects.route('projects/<int:id>', methods=['GET'])
@Auth.verify_token
def get_project(current_user, id):
    project = Project.get_user_projects(current_user).filter(Project.id == id).first()
    if project is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
    return project.to_dict()


@bp_user_projects.route('projects/<int:id>', methods=['DELETE'])
@Auth.verify_token
def delete_project(current_user, id):
    project = Project.get_user_projects(current_user).filter(Project.id == id).first()
    if project is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
    db.session.delete(project)
    db.session.commit()
    return {"message": 'Project deleted successfully'}, 200


@bp_user_projects.route('projects/<int:id>',  methods=['PATCH'])
@Auth.verify_token
def update_project(current_user, id):
    project = Project.get_user_projects(current_user).filter(Project.id == id).first()
    if project is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
    data = request.get_json(force=True)
    if data.get('name'):
        project.name = data.get('name')

    db.session.commit()
    return {"message": 'Project has been updated!!', "data": project.to_dict()}
