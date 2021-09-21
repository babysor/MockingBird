from web import webApp
from gevent import pywsgi as wsgi


if __name__ == "__main__":
    app = webApp()
    host = app.config.get("HOST")
    port = app.config.get("PORT")
    print(f"Web server: http://{host}:{port}")
    server = wsgi.WSGIServer((host, port), app)
    server.serve_forever()
