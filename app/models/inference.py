
from numpy import short
from app import db
from sqlalchemy.dialects.postgresql import JSON
from app.models.graph import Graph
from app.models.process_log import ProcessLog
from app.models.project import Project


class Inference(db.Model):
    __tablename__ = 'inference'
    id = db.Column(db.Integer, primary_key=True, unique=True)
    name = db.Column(db.String(128), index=True, nullable=False)
    model_path = db.Column(db.String(300), nullable=True)
    settings = db.Column(JSON)
    created_at = db.Column(db.DateTime(128), server_default=db.func.now())
    # Relationship
    graph_id = db.Column(db.Integer, db.ForeignKey('graph.id'))

    def __repr__(self):
        return '<Inference Name: {} >'.format(self.name)

    @classmethod
    def get_user_inferences(cls, current_user):
        return Inference.query.join(Graph).join(Project).filter(Project.company_id == current_user.company_id)

    @staticmethod
    def get_all():
        return Inference.query.all()

    @staticmethod
    def get(id):
        return Inference.query.get(int(id))

    def to_dict(self):
        process_logs = ProcessLog.query.filter(ProcessLog.graph_id == self.graph_id and ProcessLog.type == "training").all()
        data = {
            'id': self.id,
            'name': self.name,
            'settings': self.settings,
            'model_path': self.model_path,
            'created_at': self.created_at,
            'process_logs': [p.to_dict(short=True) for p in process_logs]
        }
        return data

    def get_name(self):
        data = {
            'id': self.id,
            'name': self.name,
        }
        return data


# """Dataset arguments"""
# parser.add_argument('--graph_dir', type=str, default='output/graphs/OAG_grap_reddit_10000.pk',
#                     help='The address of preprocessed graph.')
# parser.add_argument('--model_dir', type=str, default='output',
#                     help='The address for storing the models and optimization results.')
# parser.add_argument('--graph_params_dir', type=str, default='config/HGT_graph_params_Reddit.json',
#                     help='The address of the graph params file')
# parser.add_argument('--main_node', type=str, default='post',
#                     help='The name of the main node in the graph')
# parser.add_argument('--predicted_node_name', type=str, default='field',
#                     help='The name of the node that its values to be predicted')
# parser.add_argument('--edge_name', type=str, default='post_subreddit',
#                     help='The name of edge')
# parser.add_argument('--exract_attention', type=bool, default=False,
#                     help='extract the attention lists')
# parser.add_argument('--show_tensor_board', type=bool, default=False,
#                     help='show tensor board')

# """Model arguments """
# parser.add_argument('--multi_lable_task', type=bool, default=True,
#                     help='Multi label classification task')
# parser.add_argument('--conv_name', type=str, default='hgt',
#                     choices=['hgt', 'gcn', 'gat', 'rgcn', 'han', 'hetgnn'],
#                     help='The name of GNN filter. By default is Heterogeneous Graph Transformer (hgt)')
# parser.add_argument('--n_hid', type=int, default=400,
#                     help='Number of hidden dimension')
# parser.add_argument('--n_heads', type=int, default=8,
#                     help='Number of attention head')
# parser.add_argument('--n_layers', type=int, default=4,
#                     help='Number of GNN layers')
# parser.add_argument('--dropout', type=float, default=0.2,
#                     help='Dropout ratio')
# parser.add_argument('--sample_depth', type=int, default=6,
#                     help='How many numbers to sample the graph')
# parser.add_argument('--sample_width', type=int, default=128,
#                     help='How many nodes to be sampled per layer per type')

# """Optimization arguments"""
# parser.add_argument('--optimizer', type=str, default='adamw',
#                     choices=['adamw', 'adam', 'sgd', 'adagrad'],
#                     help='optimizer to use.')
# parser.add_argument('--data_percentage', type=float, default=1.0,
#                     help='Percentage of training and validation data to use')
# parser.add_argument('--n_epoch', type=int, default=30,
#                     help='Number of epoch to run')
# parser.add_argument('--n_pool', type=int, default=4,
#                     help='Number of process to sample subgraph')
# parser.add_argument('--n_batch', type=int, default=32,
#                     help='Number of batch (sampled graphs) for each epoch')
# parser.add_argument('--repeat', type=int, default=2,
#                     help='How many time to train over a singe batch (reuse data)')
# parser.add_argument('--batch_size', type=int, default=256,
#                     help='Number of output nodes for training')
# parser.add_argument('--clip', type=float, default=0.25,
#                     help='Gradient Norm Clipping')
# parser.add_argument('--include_fake_edges', type=bool, default=False,
#                     help='Include fake edges')
# parser.add_argument('--remove_edges', type=bool, default=False,
#                     help='remove weak edges')
# parser.add_argument('--restructure_at_epoch', type=int, default=10,
#                     help='restructure graph at after x epoch')
