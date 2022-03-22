import os
from flask import Blueprint, request
from app.models.dataset import DatasetFile
from app import db
from app.models.graph import Graph
from app.services.user.decorators import Auth
from werkzeug.utils import secure_filename
import pandas as pd


bp_user_datasets = Blueprint('datasets', 'dataset')

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return True
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def rename_file(name, graph_id):
    i = 0
    while(True):
        file = DatasetFile.query.filter(DatasetFile.graph_id==graph_id, DatasetFile.filename==name).first()
        if not file:
            return name
        i += 1
        name = "{}-{}.{}".format(name.rsplit('.', 1)[0], i, name.rsplit('.', 1)[1])

@bp_user_datasets.route('/datasets/files', methods=['POST'])
@Auth.verify_token
def upload_file(current_user):
    file = request.files.get('file')
    graph_id = request.form['graph_id']
    sep = request.form.get('sep', '\t')
    if not file or not allowed_file(file.filename):
        response_object = {
            'status': 'fail',
            'message': 'File extenstion is not allowed'
        }
        return response_object, 400
    filename = secure_filename(file.filename)
    filename = rename_file(filename, graph_id)
    type = request.form.get('type', filename.rsplit('.', 1)[1])

    graph = Graph.get_user_graphs(current_user).filter(Graph.id == int(graph_id)).first()
    if not graph:
        response_object = {
            'status': 'fail',
            'message': 'graph not found'
        }
        return response_object, 400
   
    path = os.path.join(
        "storage/{}/{}/dataset".format(
           graph.project_id, graph.id, graph_id)    
    )
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        os.umask(0)
        os.makedirs(path, mode=0o777)
    file.save(os.path.join(path, filename))
    
    new_file = DatasetFile(filename=filename, graph_id=graph_id, path=os.path.join(path, filename), type=type, sep=sep)
    db.session.add(new_file)
    db.session.commit()
    return {"message": 'New dataset has been created!!', "data": new_file.to_dict()}, 200
    

# Get all files of a graph
@bp_user_datasets.route('/datasets/files', methods=['GET'])
@Auth.verify_token
def get_all_datasets_files(current_user):
    graph_id = request.args.get('graph_id')
    graph = Graph.get_user_graphs(current_user).filter(Graph.id == int(graph_id)).first()
    if not graph:
        return {"status": "failed", "message": 'Graph could not be found'}, 400
    files = DatasetFile.get_user_datasets(current_user).filter(DatasetFile.graph_id==int(graph_id))
    result = [file.to_dict() for file in files]
    return {'data': result}, 200

@bp_user_datasets.route('/datasets/files/<int:id>/header', methods=['GET'])
@Auth.verify_token
def get_csv_header(current_user, id):
    try:
        file = DatasetFile.get_user_datasets(current_user).filter(DatasetFile.id == id).first()
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
def delete_dataset(current_user, id):
    file = DatasetFile.get_user_datasets(current_user).filter(DatasetFile.id == int(id)).first()
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