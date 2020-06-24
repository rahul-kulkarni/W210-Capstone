#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019 Apple Inc. All Rights Reserved.
#

import argparse
from collections import namedtuple
from pathlib import Path
import yaml

import flask

from src.run_mrqa import predict, setup_model

app = flask.Flask(__name__)

assets = None
config_args = None


def to_namedtuple(d):
    ARGS_TUPLE = namedtuple('RunArguments', sorted(d))
    return ARGS_TUPLE(**d)


def prep_model():
    global assets, config_args
    with open(str(Path(__file__).parent / 'mrqa.yaml')) as f:
        config = yaml.load(f)

    model_args = config['parameters']['predict']
    if 'model_config_overrides' in model_args:
        model_args['model_config_overrides'] = model_args['model_config_overrides'][
            model_args['model_type']
        ]
    config_args = to_namedtuple(model_args)

    config_args = config_args._replace(model_name_or_path=str(Path(__file__).parent / 'model'))
    assets = setup_model(config_args)


def _predict(test_data):
    global assets, config_args
    results = predict(config_args, assets, test_data, include_labels=False)

    predictions = results['predictions']

    return predictions


@app.route('/', methods=['POST'])
def index():
    json_obj = flask.request.get_json()
    pred = _predict(json_obj)
    return flask.jsonify(pred)


if __name__ == '__main__':
    parse = argparse.ArgumentParser('')
    parse.add_argument('port', type=int)
    args = parse.parse_args()

    prep_model()

    app.run(port=args.port)
