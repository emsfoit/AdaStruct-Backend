from flask import Blueprint, request
from app.models.process_log import ProcessLog
from app import db
from app.models.process_log import ProcessLog
from app.services.user.decorators import Auth
bp_user_process_logs = Blueprint('process_logs', 'process_log')


@bp_user_process_logs.route('/process_logs', methods=['GET'])  # return all process_logs
@Auth.verify_token
def get_all_process_logs(current_user):
    process_logs = ProcessLog.get_user_process_logs(current_user)
    graph_id = request.args.get('graph_id', None)
    if graph_id:
        process_logs = process_logs.filter(ProcessLog.graph_id == int(graph_id))
    type = request.args.get('type', None)
    if type:
        process_logs = process_logs.filter(ProcessLog.type == type)
    result = [process_log.to_dict(short=True) for process_log in process_logs.all()]
    return {'data': result}, 200


@bp_user_process_logs.route('process_logs/<int:id>', methods=['GET'])
@Auth.verify_token
def get_process_log(current_user, id):
    process_log = ProcessLog.get_user_process_logs(current_user).filter(ProcessLog.id==id).first()
    if process_log is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
    return process_log.to_dict()


@bp_user_process_logs.route('process_logs/<int:id>', methods=['DELETE'])
@Auth.verify_token
def delete_process_log(current_user, id):
    process_log = ProcessLog.get_user_process_logs(current_user).filter(ProcessLog.id==id).first()
    if process_log is None:
        response_object = {
            'status': 'fail',
            'message': 'not found.'
        }
        return response_object, 404
    db.session.delete(process_log)
    db.session.commit()
    return {"message": 'ProcessLog deleted successfully'}, 200

