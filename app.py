"""Приложение Flask для защиты ВКР. Flask и его зависимости входят в комплект."""
import argparse
import sys
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / '_vendor'))
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.serving import make_server
from predictor import load_bundle, predict_request


def create_app(bundle=None):
    bundle = bundle or load_bundle()
    app = Flask(__name__, template_folder='web', static_folder=None)
    app.config['MAX_CONTENT_LENGTH'] = 32000

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/schema')
    def schema():
        return jsonify(features=bundle['features'], ratio_features=bundle['ratio_features'], results=bundle['results'])

    @app.get('/api/metrics')
    def metrics():
        return jsonify(bundle['results'])

    @app.get('/figures/<filename>')
    def figure(filename):
        allowed = {'histograms.png', 'boxplots.png', 'correlation_heatmap.png', 'scatter_matrix.png',
                   'normalization_before_after.png', 'model_comparison.png',
                   'actual_predicted_modulus.png', 'actual_predicted_strength.png', 'neural_learning_curve.png'}
        if filename not in allowed:
            return jsonify(error='Страница не найдена'), 404
        return send_from_directory(ROOT / 'reports' / 'figures', filename)

    @app.post('/api/predict')
    def predict():
        try:
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                raise ValueError('Некорректный запрос')
            result = predict_request(bundle, payload.get('mode'), payload.get('values'))
            return jsonify(prediction=result)
        except (ValueError, TypeError) as e:
            return jsonify(error=str(e)), 400

    @app.errorhandler(404)
    def missing(error):
        return jsonify(error='Страница не найдена'), 404

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error='Слишком большой запрос'), 413

    @app.after_request
    def no_cache(response):
        response.headers['Cache-Control'] = 'no-store'
        return response

    return app


def main():
    parser = argparse.ArgumentParser(description='Flask: прогнозирование свойств композитов')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    try:
        app = create_app()
    except FileNotFoundError:
        print('Нет файла models/portable_models.json. Распакуйте архив полностью либо выполните train.py.')
        return 1
    try:
        server = make_server('127.0.0.1', args.port, app, threaded=True)
    except (OSError, SystemExit):
        print('Порт занят. Закройте предыдущее окно приложения или запустите: python app.py --port 8766')
        return 1
    url = f'http://127.0.0.1:{args.port}'
    print('Приложение Flask запущено:', url, flush=True)
    print('Оставьте это окно открытым. Для остановки нажмите Ctrl+C.', flush=True)
    if not args.no_browser:
        threading.Timer(.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
