
import base64

from flask import Flask, request, render_template, Response
from flask_cors import CORS
from gevent import pywsgi as wsgi
from flask_wtf import CSRFProtect


def server():
    app = Flask(__name__)
    app.config['MAX_CONTENT_PATH'] = 1024 * 1024 * 4 # mp3文件大小限定不能超过4M
    app.config['SECRET_KEY'] = "mockingbird_key"
    app.config['WTF_CSRF_SECRET_KEY'] = "mockingbird_key"

    #CORS(app) #允许跨域，注释掉此行则禁止跨域请求

    csrf = CSRFProtect(app)
    csrf.init_app(app)

    @app.route('/', methods=['GET'])
    def index():
        return render_template("index.html")

    @csrf.exempt
    @app.route('/api/synthesize', methods=['POST'])
    def api_synthesize():
        if request.method == 'POST':
            text = request.form["text"] #获取用户输入的文本
            wav_base64 = request.form["upfile_b64"]
            wav = base64.b64decode(wav_base64)
            if len(wav) < 1024:
                wav = None #用户上传wav过短，此处使用服务端本地默认wav文件

            wav = wav # 将此处替换为处理音频的函数

            # f = open('D:/mockingbird/upload/' + text + '.wav', 'wb')
            # f.write(wav)
            # f.close()
            
            #返回结果，wav格式
            return Response(wav, mimetype='audio/wav')
    server = wsgi.WSGIServer(("0.0.0.0", 9494), app)
    server.serve_forever()

if __name__ == '__main__':
    server()
