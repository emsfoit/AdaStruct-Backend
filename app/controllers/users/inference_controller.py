from flask import Blueprint, request, send_file
from app import db
from app.models.process_log import ProcessLog
from app.services.user.decorators import Auth
import os
from datetime import datetime
import multiprocess as mp
from multiprocessing import Process
from flask import current_app
from app.models.graph import Graph
from app.models.inference import Inference
import subprocess

bp_user_inferences = Blueprint('inferences', 'inference')


@bp_user_inferences.route('/inferences', methods=['POST'])
@Auth.verify_token
def create_inference(current_user):
    data = request.get_json(force=True)
    name = data.get('name', None)
    graph_id = data.get('graph_id', -1)
    settings = data.get('settings', None)
    response_object = {}
    error = False
    if not name:
        response_object["name"] = ["name is invalid"]
        error = True
    graph = Graph.get_user_graphs(current_user).filter(Graph.id == int(graph_id)).first()
    if not graph:
        error = True
        response_object["graph"] = ["graph is invalid"]
    if error:
        return response_object, 400
    inference = Inference.get_user_inferences(current_user).filter(Inference.graph_id==int(graph_id), Graph.name==name).first()
    if not inference:
        new_inference = Inference(name=name, graph_id=int(graph_id), settings=settings)
        db.session.add(new_inference)
        db.session.commit()
        return {"message": 'New inference has been created!!', "data": new_inference.to_dict()}
    response_object = {
        "name": ["name is invalid"]
    }
    return response_object, 400
    

@bp_user_inferences.route('/inferences', methods=['GET'])  # return all inferences
@Auth.verify_token
def get_all_inferences(current_user):
    inferences = Inference.get_user_inferences(current_user).all()
    result = [inference.to_dict() for inference in inferences]
    return {'data': result}, 200


@bp_user_inferences.route('inferences/<int:id>', methods=['GET'])
@Auth.verify_token
def get_inference(current_user, id):
    inference = Inference.get_user_inferences(current_user).filter(Inference.id==id).first()
    if inference is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
    return inference.to_dict()


@bp_user_inferences.route('inferences/<int:id>', methods=['DELETE'])
@Auth.verify_token
def delete_inference(current_user, id):
    inference = Inference.get_user_inferences(current_user).filter(Inference.id==id).first()
    if inference is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
    db.session.delete(inference)
    db.session.commit()
    return {"message": 'Inference deleted successfully'}, 200


@bp_user_inferences.route('inferences/<int:id>',  methods=['PATCH'])
@Auth.verify_token
def update_inference(current_user, id):
    inference =  Inference.get_user_inferences(current_user).filter(Inference.id==id).first()
    if inference is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
 
    data = request.get_json(force=True)
    if data.get('settings'):
        inference.settings = data.get('settings')
    if data.get('name'):
        inference.name = data.get('name')
    db.session.commit()

    return {"message": 'Inference has been updated!!', "data": inference.to_dict()}


# Download the Model if exist
@bp_user_inferences.route('inferences/<int:id>/download',  methods=['GET'])
@Auth.verify_token
def download_inference(current_user, id):
    inference = Inference.get_user_inferences(current_user).filter(Inference.id==id).first()
    if inference is None or not inference.model_path:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
    path =  os.path.join(os.path.dirname(current_app.instance_path), inference.model_path)
    return send_file(path, as_attachment=True)



@bp_user_inferences.route('inferences/<int:id>/train',  methods=['get'])
@Auth.verify_token
def train(current_user, id):
    inference =  Inference.get_user_inferences(current_user).filter(Inference.id==id).first()
    if inference is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404

    if True:
        process_log = ProcessLog(name="test", type="training", graph_id=inference.graph_id, status="Init", log="")
        db.session.add(process_log)
        db.session.commit()
        heavy_process = Process(
            target=run_inference,
            args=(current_user, inference, process_log),
            daemon=True
        )
        heavy_process.start()
        process_log.status = "training,{}".format(heavy_process.pid)
        db.session.commit()
        # run_inference(current_user, inference)
    return {"message": 'Training will start soon!', "data": inference.to_dict()}

# build inference process
def run_inference(current_user, inference, process_log):
    graph = Graph.get(inference.id)
    # storage/project_id/graph_id/model/training_id/
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    output_inference_file = os.path.join(
        "storage/{}/{}/model/{}/{}".format(
            graph.project_id, graph.id, inference.id, dt_string)    
    )
    # Check whether the specified path exists or not
    isExist = os.path.exists(output_inference_file)
    if not isExist:
        os.umask(0)
        os.makedirs(output_inference_file, mode=0o777)

    output_inference_file=os.path.join(output_inference_file, 'model')


    try:
        subprocess.call(['python', '-m', 'tools.hgt.training.main', '-inference_id', "{}".format(inference.id), '-process_log_id',"{}".format(process_log.id)])
    except Exception as e:
        process_log.add_to_log("Failed")
        process_log.status = "failed"
        db.session.commit()
    inference.modal_path = output_inference_file
    process_log.add_to_log("process completed!")
    process_log.status = "completed"
    db.session.commit()


# kill a process
# TODO: Find a solution to kill a procees (Not working) 
@bp_user_inferences.route('inferences/<int:id>/kill',  methods=['post'])
@Auth.verify_token
def kill(current_user, id):
    inference = Inference.get(id)
    data = request.get_json(force=True)
    process_id = data.get('process_id', None)
    process_log = ProcessLog.get(process_id)
    if not process_log:
        return {"message": 'Process Not found'}, 400
    try:
        101140
        p = [process for process in mp.active_children() if Process.pid == 101140]
        p.terminate()
        process_log.add_to_log("Terminated by the user")
        process_log.status = "Terminated"
        db.session.commit()
        return {"message": 'Process Terminated correctly!!', "data": inference.to_dict()}
    except Exception as e:
        return {"message": 'Could not terminate the process'}, 400



