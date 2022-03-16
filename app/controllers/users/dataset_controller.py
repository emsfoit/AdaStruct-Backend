import os
from unittest.mock import patch
from flask import Blueprint, request
from app.models.dataset import DatasetFile
from app import db, app
from app.models.project import Project
from app.services.user.decorators import Auth
from werkzeug.utils import secure_filename
import pandas as pd


bp_user_datasets = Blueprint('datasets', 'dataset')

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return True
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def rename_file(name, project_id):
    i = 0
    while(True):
        file = DatasetFile.query.filter(DatasetFile.project_id==project_id, DatasetFile.filename==name).first()
        if not file:
            return name
        i += 1
        name = "{}-{}.{}".format(name.rsplit('.', 1)[0], i, name.rsplit('.', 1)[1])

@bp_user_datasets.route('/datasets/files', methods=['POST'])
@Auth.verify_token
def upload_file(current_user):
    file = request.files.get('file')
    project_id = request.form['project_id']
    sep = request.form.get('sep', '\t')


    if not file or not allowed_file(file.filename):
        response_object = {
            'status': 'fail',
            'message': 'File extenstion is not allowed'
        }
        return response_object, 400
    filename = secure_filename(file.filename)
    filename = rename_file(filename, project_id)
    type = request.form.get('type', filename.rsplit('.', 1)[1])
    path = os.path.join(
        "{}/{}/{}".format(
            app.config['UPLOAD_FOLDER'], current_user.company_id, project_id)    
    )
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        os.umask(0)
        os.makedirs(path, mode=0o777)
    file.save(os.path.join(path, filename))
    
    new_file = DatasetFile(filename=filename, project_id=project_id, path=os.path.join(path, filename), type=type, sep=sep)
    db.session.add(new_file)
    db.session.commit()
    return {"message": 'New dataset has been created!!', "data": new_file.to_dict()}, 200
    

# Get all files of a project
@bp_user_datasets.route('/datasets/files', methods=['GET'])
@Auth.verify_token
def get_all_datasets_files(current_user):
    project_id = request.args.get('project_id')
    if not project_id:
        return {"status": "failed", "message": 'Project id not found'}, 400
    project = Project.get(project_id)
    if project.company_id != current_user.company_id:
        response_object = {
            'status': 'fail',
            'message': 'Not Allowed.'
        }
        return response_object, 500

    files = DatasetFile.query.filter(DatasetFile.project_id==project_id)
    result = [file.to_dict() for file in files]
    return {'data': result}, 200

@bp_user_datasets.route('/datasets/files/<int:file_id>/header', methods=['GET'])
@Auth.verify_token
def get_csv_header(current_user, file_id):
    try:
        file = DatasetFile.get(file_id)
        columns = list(pd.read_csv(file.path, sep="\t").columns)
        return {'data': columns}, 200
    except:
        response_object = {
            'status': 'fail',
            'message': 'Could not read File'
        }
        return response_object, 500



@bp_user_datasets.route('/datasets/files/<int:id>', methods=['DELETE'])
@Auth.verify_token
def delete_project(current_user, id):
    file = DatasetFile.get(id)
    if file is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
    
    os.remove(file.path)
    db.session.delete(file)
    db.session.commit()
    return {"message": 'file deleted successfully'}, 200