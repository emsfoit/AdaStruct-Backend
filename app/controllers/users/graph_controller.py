from flask import Blueprint, request, send_file
from app.models.graph import Graph
from app import db
from app.models.project import Project
from app.models.process_log import ProcessLog

from app.services.user.decorators import Auth
from tools.hgt.graph_builder.build_HGT_graph_OAG import build_hgt_graph
import pandas as pd
import os
from datetime import datetime
from multiprocessing import Process
from flask import current_app

from tools.hgt.utils.utils import logger

bp_user_graphs = Blueprint('graphs', 'graph')


@bp_user_graphs.route('/graphs', methods=['POST'])
@Auth.verify_token
def create_graph(current_user):
    data = request.get_json(force=True)
    name = data.get('name', None)
    project_id = data.get('project_id', None)
    settings = data.get('settings', None)
    response_object = {}
    error = False
    if not name:
        response_object["name"] = ["name is invalid"]
        error = True
    project = Project.get_user_projects(current_user).filter(Project.id == int(project_id)).first()
    if not project:
        error = True
        response_object["project"] = ["project is invalid"]
    if error:
        return response_object, 400
    graph = Graph.query.filter(Graph.project_id==int(project_id), Graph.name==name).first()
    if not graph:
        new_graph = Graph(name=name, project_id=int(project_id), settings=settings)
        db.session.add(new_graph)
        db.session.commit()
        return {"message": 'New graph has been created!!', "data": new_graph.to_dict()}
    response_object = {
        "name": ["name is invalid"]
    }
    return response_object, 400
    


@bp_user_graphs.route('/graphs', methods=['GET'])  # return all graphs
@Auth.verify_token
def get_all_graphs(current_user):
    graphs = Graph.get_users_graph(current_user).all()
    result = [graph.to_dict() for graph in graphs]
    return {'data': result}, 200


@bp_user_graphs.route('graphs/<int:id>', methods=['GET'])
@Auth.verify_token
def get_graph(current_user, id):
    graph = Graph.get_users_graph(current_user).filter(Graph.id==id).first()
    if graph is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
    return graph.to_dict()


@bp_user_graphs.route('graphs/<int:id>', methods=['DELETE'])
@Auth.verify_token
def delete_graph(current_user, id):
    graph = Graph.get_users_graph(current_user).filter(Graph.id==id).first()
    if graph is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
    db.session.delete(graph)
    db.session.commit()
    return {"message": 'Graph deleted successfully'}, 200


@bp_user_graphs.route('graphs/<int:id>',  methods=['PATCH'])
@Auth.verify_token
def update_graph(current_user, id):
    graph = Graph.get_users_graph(current_user).filter(Graph.id==id).first()
    if graph is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
 
    data = request.get_json(force=True)
    if data.get('settings'):
        graph.settings = data.get('settings')
    if data.get('name'):
        graph.name = data.get('name')

    db.session.commit()
    if True:
        heavy_process = Process(
            target=build_graph,
            args=(current_user,graph),
            daemon=True
        )
        heavy_process.start()
        # build_graph(current_user, graph)
    return {"message": 'Graph has been updated!!', "data": graph.to_dict()}


# build graph process
def build_graph(current_user, graph):
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    output_graph_file = os.path.join(
        "storage/{}/{}/graph/{}".format(
            graph.project_id, graph.id ,dt_string)    
    )
    # Check whether the specified path exists or not
    isExist = os.path.exists(output_graph_file)
    if not isExist:
        os.umask(0)
        os.makedirs(output_graph_file, mode=0o777)

    data = {}
    for f in graph.files:
        df = pd.read_csv(f.path, sep=f.sep)
        data[f.filename] = df
    output_graph_file=os.path.join(output_graph_file, 'graph.pk')
    process_log = ProcessLog(name="test", type="build", graph_id= graph.id, status="started", log="")
    db.session.add(process_log)
    db.session.commit()
    def logger(msg):
        print(msg)
        process_log.add_to_log(msg)
    result =  build_hgt_graph(graph.settings, data=data, output_graph_file=output_graph_file, logger=logger)
    if result:
        graph.path = output_graph_file
        logger("process completed!")
        process_log.status = "completed"
        db.session.commit()
    else:
        logger("process failed!")
        process_log.status = "failed"


@bp_user_graphs.route('graphs/<int:id>/download',  methods=['GET'])
@Auth.verify_token
def download_graph(current_user, id):
    graph = Graph.get_users_graph(current_user).filter(Graph.id==id).first()
    if graph is None or not graph.path:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
    path =  os.path.join(os.path.dirname(current_app.instance_path), graph.path)
    return send_file(path, as_attachment=True)