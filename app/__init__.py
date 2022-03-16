from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from config import BaseConfig as Config
from flask_migrate import Migrate
from flask_cors import CORS

db = SQLAlchemy()
bcrypt = Bcrypt()
login_manager = LoginManager()
app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)
bcrypt.init_app(app)
login_manager.init_app(app)
app = Flask(__name__)
app.config.from_object(Config)


db.init_app(app)
migrate = Migrate(app, db)
bcrypt.init_app(app)
login_manager.init_app(app)

# User
from app.models import user
from app.models import project
from app.models import company
from app.models import dataset



cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
from app.controllers.users.account_controller import bp_user_account
app.register_blueprint(bp_user_account, url_prefix='/api/users')


from app.controllers.users.projects_controller import bp_user_projects
app.register_blueprint(bp_user_projects, url_prefix='/api/users')

from app.controllers.users.dataset_controller import bp_user_datasets
app.register_blueprint(bp_user_datasets, url_prefix='/api/users')

UPLOAD_FOLDER = 'dataset'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER