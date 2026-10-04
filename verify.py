"""Проверка приложения и переносимых прогнозов, без сторонних библиотек."""
import json
import math
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from app import create_app
from werkzeug.serving import make_server
from predictor import load_bundle, predict_request


def main():
    root = Path(__file__).resolve().parent
    bundle = load_bundle()
    cases = json.loads((root / 'reports' / 'verification_cases.json').read_text(encoding='utf-8'))
    server = make_server('127.0.0.1', 0, create_app(bundle), threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = 'http://127.0.0.1:' + str(server.server_port)
    checks = 0
    def request(path, obj=None):
        req = Request(base + path, data=None if obj is None else json.dumps(obj).encode('utf-8'),
                      headers={} if obj is None else {'Content-Type': 'application/json'})
        return urlopen(req, timeout=10)
    try:
        for path in ['/', '/api/schema', '/api/metrics', '/figures/model_comparison.png']:
            with request(path) as response:
                assert response.status == 200 and response.read()
                checks += 1
        for case in cases:
            direct = predict_request(bundle, case['mode'], case['values'])
            with request('/api/predict', {'mode': case['mode'], 'values': case['values']}) as response:
                http = json.loads(response.read())['prediction']
            for name, expected in case['expected'].items():
                tolerance = 2e-5 if case['mode'] == 'ratio' else 1e-8
                relative = 1e-6 if case['mode'] == 'ratio' else 1e-10
                assert math.isclose(direct[name], expected, rel_tol=relative, abs_tol=tolerance)
                assert math.isclose(http[name], expected, rel_tol=relative, abs_tol=tolerance)
                checks += 2
        defaults = {f['name']: f['default'] for f in bundle['features']}
        first = bundle['features'][0]['name']
        invalid = [('unknown', defaults), ('properties', {}),
                   ('properties', {**defaults, first: float('nan')}),
                   ('properties', {**defaults, first: -1}),
                   ('properties', {**defaults, 'Угол нашивки, град': 45})]
        for mode, values in invalid:
            try:
                request('/api/predict', {'mode': mode, 'values': values})
                raise AssertionError('Ожидалась ошибка HTTP 400')
            except HTTPError as e:
                assert e.code == 400
                assert 'error' in json.loads(e.read())
                checks += 1
        try:
            request('/missing-page')
            raise AssertionError('Ожидалась ошибка HTTP 404')
        except HTTPError as e:
            assert e.code == 404
            checks += 1
        result = {'status': 'passed', 'checks': checks, 'comparison': 'Прогнозы sklearn/Keras и JSON через Flask',
                  'dependencies': 'Flask и зависимости включены в _vendor; установка пользователем не нужна'}
        (root / 'reports' / 'app_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Проверка пройдена. Количество проверок:', checks)
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
