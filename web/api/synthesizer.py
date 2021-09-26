from pathlib import Path
from flask_restx import Namespace, Resource, fields

api = Namespace('synthesizers', description='Synthesizers related operations')

synthesizer = api.model('Synthesizer', {
    'name': fields.String(required=True, description='The synthesizer name'),
    'path': fields.String(required=True, description='The synthesizer path'),
})

synthesizers_cache = {}
syn_models_dirt = "synthesizer/saved_models"
synthesizers = list(Path(syn_models_dirt).glob("**/*.pt"))
print("Loaded synthesizer models: " + str(len(synthesizers)))

@api.route('/')
class SynthesizerList(Resource):
    @api.doc('list_synthesizers')
    @api.marshal_list_with(synthesizer)
    def get(self):
        '''List all synthesizers'''
        return list({"name": e.name, "path": str(e)} for e in synthesizers)

