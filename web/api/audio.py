import os
from pathlib import Path
from flask_restx import Namespace, Resource, fields
from flask import Response, current_app

api = Namespace('audios', description='Audios related operations')

audio = api.model('Audio', {
    'name': fields.String(required=True, description='The audio name'),
})

def generate(wav_path):
    with open(wav_path, "rb") as fwav:
        data = fwav.read(1024)
        while data:
            yield data
            data = fwav.read(1024)

@api.route('/')
class AudioList(Resource):
    @api.doc('list_audios')
    @api.marshal_list_with(audio)
    def get(self):
        '''List all audios'''
        audio_samples = []
        AUDIO_SAMPLES_DIR = current_app.config.get("AUDIO_SAMPLES_DIR")
        if os.path.isdir(AUDIO_SAMPLES_DIR):
            audio_samples = list(Path(AUDIO_SAMPLES_DIR).glob("*.wav"))
        return list(a.name for a in audio_samples)

@api.route('/<name>')
@api.param('name', 'The name of audio')
@api.response(404, 'audio not found')
class Audio(Resource):
    @api.doc('get_audio')
    @api.marshal_with(audio)
    def get(self, name):
        '''Fetch a cat given its identifier'''
        AUDIO_SAMPLES_DIR = current_app.config.get("AUDIO_SAMPLES_DIR")
        if Path(AUDIO_SAMPLES_DIR + name).exists():
            return Response(generate(AUDIO_SAMPLES_DIR + name), mimetype="audio/x-wav")
        api.abort(404)
    