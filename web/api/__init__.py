from flask import Blueprint
from flask_restx import Api
from .audio import api as audio

api_blueprint = Blueprint('api', __name__, url_prefix='/api')

api = Api(
    app=api_blueprint,
    title='Mocking Bird',
    version='1.0',
    description='My API'
)

api.add_namespace(audio)
