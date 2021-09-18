import os
from pathlib import Path
from gevent import pywsgi as wsgi
from flask import Flask, jsonify, Response

# Init and load config
app = Flask(__name__, instance_relative_config=True)

app.config.from_object("config.default")

# API For Non-Trainer
# 1. list sample audio files
# 2. record / upload / select audio files
# 3. load melspetron of audio
# 4. inference by audio + text + models(encoder, vocoder, synthesizer)
# 5. export result
audio_samples = []
AUDIO_SAMPLES_DIR = app.config.get("AUDIO_SAMPLES_DIR")
if os.path.isdir(AUDIO_SAMPLES_DIR):
    audio_samples = list(Path(AUDIO_SAMPLES_DIR).glob("*.wav"))
print("Loaded samples: " + str(len(audio_samples)))
enc_models_dir = "encoder/saved_models"
syn_models_dirt = "synthesizer/saved_models"
voc_models_di = "vocoder/saved_models"
encoder = list(Path(enc_models_dir).glob("*.pt"))
vocoder = list(Path(voc_models_di).glob("**/*.pt"))
synthesizer = list(Path(syn_models_dirt).glob("**/*.pt"))
print("Loaded encoder models: " + str(len(encoder)))
print("Loaded vocoder models: " + str(len(vocoder)))
print("Loaded synthesizer models: " + str(len(synthesizer)))

@app.route("/audios", methods=["GET"])
def audios():
    return jsonify(
        {"data": list(a.name for a in audio_samples), "total": len(audio_samples)}
    )

@app.route("/audios/<name>", methods=["GET"])
def audio_play(name):
    def generate():
        with open(AUDIO_SAMPLES_DIR + name, "rb") as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
    return Response(generate(), mimetype="audio/x-wav")

@app.route("/models", methods=["GET"])
def models():
    return jsonify(
        {
            "encoder": list(e.name for e in encoder),
            "vocoder": list(e.name for e in vocoder),
            "synthesizer": list(s.name for s in synthesizer),
        }
    )

# Cache for synthesizer
@app.route("/synthesizer", methods=["GET"])
def synthesizer():
    # TODO Implementation
    return 

# API For Trainer / Pro User
# TODO Implementation
@app.route("/", methods=["GET"])
def root():
    return jsonify({"msg": "This is the root of Mocking bird server"})

if __name__ == "__main__":
    host = app.config.get("HOST")
    port = app.config.get("PORT")
    print(f"Web server: http://{host}:{port}")
    server = wsgi.WSGIServer((host, port), app)
    server.serve_forever()
