# -*- coding: utf-8 -*-


from web.gif_classify_service import route_gif
from flask_cors import *
from flask import Flask

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.register_blueprint(route_gif, url_prefix='/gif')

if __name__ == "__main__":
    app.run('0.0.0.0', port=9527)
