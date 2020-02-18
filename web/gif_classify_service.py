# coding=utf8

import json
from flask import request, Blueprint, Flask
from web import config
from predictor import Predictor

route_gif = Blueprint('gif_page', __name__)

# do initial
predictor = Predictor(config)


@route_gif.route('/classify', methods=['POST', 'GET'])
def get_classify():
    video_path = request.json.get("f_path")
    result = predictor.predict(video_path)
    return json.dumps(result, ensure_ascii=False)

