from concurrent.futures import process
from app import db
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSON
from app.models.graph import Graph
from app.models.project import Project

# will be used to store training performance at each epoch
class ExtraInfo(db.Model):
    __tablename__ = 'extra_info'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    data = db.Column(JSON)
    created_at = db.Column(db.DateTime(128), server_default=db.func.now())
    # Relationship
    process_log_id = db.Column(db.Integer, db.ForeignKey('process_log.id'))
    
    def to_dict(self):
        data = {
            'data': self.data,
            'created_at': self.created_at
        }
        return data

class ProcessLog(db.Model):
    __tablename__ = 'process_log'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    name = db.Column(db.String(200), index=True, nullable=False)
    # build, train
    type = db.Column(db.String(10), index=True, nullable=False)
    status = db.Column(db.String(20), index=True, nullable=False)
    log = db.Column(db.Text, index=True, nullable=True)
    created_at = db.Column(db.DateTime(128), server_default=db.func.now())
    # Relationship
    extra_info = db.relationship('ExtraInfo', backref='process_log', lazy=True, cascade="all,delete")
    graph_id = db.Column(db.Integer, db.ForeignKey('graph.id'))

    def __repr__(self):
        return '<ProcessLog Name: {} >'.format(self.name)

    @classmethod
    def get_user_process_logs(cls, current_user):
        return ProcessLog.query.join(Graph).join(Project).filter(Project.company_id == current_user.company_id)

    @staticmethod
    def get_all():
        return ProcessLog.query.all()

    @staticmethod
    def get(id):
        return ProcessLog.query.get(int(id))

    @staticmethod
    def clean():
        logs =  ProcessLog.query.filter(ProcessLog.status.contains('running')).all()
        for log in logs:
            log.status = "terminated"
        db.session.commit()

    def to_dict(self, short=False):
        data = {
            'id': self.id,
            'name': self.name,
            'type': self.type,
            'status': self.status,
            'created_at': self.created_at
        }
        if not short:
            data['extra_info'] = [elm.to_dict() for elm in self.extra_info]
            data['log'] =  self.log
        return data

    def add_to_log(self, message, print_m=False):
        final_m = "[{}] {}\n".format(
            datetime.now().strftime('%d.%m.%Y %H:%M:%S'),
            message)
        self.log += final_m
        if print_m:
            print(final_m)
        db.session.commit()

    def add_to_extra_info(self, message, print_m=False):
        if print_m:
            print(message)
        new_info = ExtraInfo(data=message, process_log_id=self.id)
        db.session.add(new_info)
        db.session.commit()