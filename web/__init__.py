from web.api import api_blueprint
from pathlib import Path
from gevent import pywsgi as wsgi
from flask import Flask, Response, request, render_template
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder.hifigan import inference as gan_vocoder
from vocoder.wavernn import inference as rnn_vocoder
import numpy as np
import re
from scipy.io.wavfile import write
import librosa
import io
import base64
from flask_cors import CORS
from flask_wtf import CSRFProtect
import webbrowser

def webApp():
    # Init and load config
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object("web.config.default")
    app.config['RESTPLUS_MASK_SWAGGER'] = False
    app.register_blueprint(api_blueprint)

    # CORS(app) #允许跨域，注释掉此行则禁止跨域请求
    csrf = CSRFProtect(app)
    csrf.init_app(app)
   
    syn_models_dirt = "synthesizer/saved_models"
    synthesizers = list(Path(syn_models_dirt).glob("**/*.pt"))
    synthesizers_cache = {}
    encoder.load_model(Path("encoder/saved_models/pretrained.pt"))
    rnn_vocoder.load_model(Path("vocoder/saved_models/pretrained/pretrained.pt"))
    gan_vocoder.load_model(Path("vocoder/saved_models/pretrained/g_hifigan.pt"))

    def pcm2float(sig, dtype='float32'):
        """Convert PCM signal to floating point with a range from -1 to 1.
        Use dtype='float32' for single precision.
        Parameters
        ----------
        sig : array_like
            Input array, must have integral type.
        dtype : data type, optional
            Desired (floating point) data type.
        Returns
        -------
        numpy.ndarray
            Normalized floating point data.
        See Also
        --------
        float2pcm, dtype
        """
        sig = np.asarray(sig)
        if sig.dtype.kind not in 'iu':
            raise TypeError("'sig' must be an array of integers")
        dtype = np.dtype(dtype)
        if dtype.kind != 'f':
            raise TypeError("'dtype' must be a floating point type")

        i = np.iinfo(sig.dtype)
        abs_max = 2 ** (i.bits - 1)
        offset = i.min + abs_max
        return (sig.astype(dtype) - offset) / abs_max

    # Cache for synthesizer
    @csrf.exempt
    @app.route("/api/synthesize", methods=["POST"])
    def synthesize():
        # TODO Implementation with json to support more platform
        # Load synthesizer
        if "synt_path" in request.form:
            synt_path = request.form["synt_path"]
        else:
            synt_path = synthesizers[0]
            print("NO synthsizer is specified, try default first one.")
        if synthesizers_cache.get(synt_path) is None:
            current_synt = Synthesizer(Path(synt_path))
            synthesizers_cache[synt_path] = current_synt
        else:
            current_synt = synthesizers_cache[synt_path]
        print("using synthesizer model: " + str(synt_path))
        # Load input wav
        if "upfile_b64" in request.form:
            wav_base64 = request.form["upfile_b64"]
            wav = base64.b64decode(bytes(wav_base64, 'utf-8'))
            wav = pcm2float(np.frombuffer(wav, dtype=np.int16), dtype=np.float32)
            sample_rate = Synthesizer.sample_rate
        else:
            wav, sample_rate,  = librosa.load(request.files['file'])
        write("temp.wav", sample_rate, wav) #Make sure we get the correct wav
        
        encoder_wav = encoder.preprocess_wav(wav, sample_rate)
        embed, _, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
        
        # Load input text
        texts = request.form["text"].split("\n")
        punctuation = '！，。、,' # punctuate and split/clean text
        processed_texts = []
        for text in texts:
            for processed_text in re.sub(r'[{}]+'.format(punctuation), '\n', text).split('\n'):
                if processed_text:
                    processed_texts.append(processed_text.strip())
        texts = processed_texts

        # synthesize and vocode
        embeds = [embed] * len(texts)
        specs = current_synt.synthesize_spectrograms(texts, embeds)
        spec = np.concatenate(specs, axis=1)
        if "vocoder" in request.form and request.form["vocoder"] == "WaveRNN":
            wav = rnn_vocoder.infer_waveform(spec)
        else:
            wav = gan_vocoder.infer_waveform(spec)

        # Return cooked wav
        out = io.BytesIO()
        write(out, Synthesizer.sample_rate, wav.astype(np.float32))
        return Response(out, mimetype="audio/wav")

    @app.route('/', methods=['GET'])
    def index():
        return render_template("index.html")

    host = app.config.get("HOST")
    port = app.config.get("PORT")
    web_address = 'http://{}:{}'.format(host, port)
    print(f"Web server:" + web_address)
    webbrowser.open(web_address)
    server = wsgi.WSGIServer((host, port), app)
    server.serve_forever()
    return app

if __name__ == "__main__":
    webApp()