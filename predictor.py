"""Прогноз по JSON: только стандартная библиотека Python, без pickle и sklearn."""
import json
import math
import struct
from pathlib import Path


def load_bundle(path=None):
    path = path or Path(__file__).resolve().parent / 'models' / 'portable_models.json'
    return json.loads(Path(path).read_text(encoding='utf-8'))


def predict_model(model, values):
    a = [float(x) for x in values]
    if model['kind'] == 'linear':
        z = [x * s + m for x, s, m in zip(a, model['factor'], model['offset'])]
        return model['intercept'] + sum(x * c for x, c in zip(z, model['coef']))
    if model['kind'] in ('forest', 'boosting'):
        # sklearn приводит вход деревьев к float32: повторяем для совпадения на порогах.
        a = [struct.unpack('f', struct.pack('f', x))[0] for x in a]
        results = []
        for t in model['trees']:
            i = 0
            while t['left'][i] != -1:
                i = t['left'][i] if a[t['feature'][i]] <= t['threshold'][i] else t['right'][i]
            results.append(t['value'][i])
        if model['kind'] == 'boosting':
            return model['init'] + model['learning_rate'] * sum(results)
        return sum(results) / len(results)
    if model['kind'] == 'mlp':
        a = [x * s + m for x, s, m in zip(a, model['factor'], model['offset'])]
        for layer_number, (weights, biases) in enumerate(zip(model['weights'], model['biases'])):
            a = [b + sum(a[i] * weights[i][j] for i in range(len(a))) for j, b in enumerate(biases)]
            if layer_number < len(model['weights']) - 1:
                a = [max(0.0, x) for x in a]
        return (a[0] - model['target_offset']) / model['target_factor']
    raise ValueError('Неизвестный тип модели')


def validated_values(features, data):
    if not isinstance(data, dict):
        raise ValueError('Ожидаются именованные поля JSON')
    values = []
    for f in features:
        if f['name'] not in data:
            raise ValueError('Заполните поле: ' + f['label'])
        try:
            value = float(str(data[f['name']]).replace(',', '.'))
        except (ValueError, TypeError):
            raise ValueError('Требуется число: ' + f['label']) from None
        if not math.isfinite(value):
            raise ValueError('Требуется конечное число: ' + f['label'])
        if value < f['min'] or value > f['max']:
            raise ValueError('Значение вне диапазона датасета: ' + f['label'])
        if f.get('choices') and value not in f['choices']:
            raise ValueError('Выберите 0 или 90 градусов')
        values.append(value)
    return values


def predict_request(bundle, mode, data):
    if mode == 'properties':
        a = validated_values(bundle['features'], data)
        return {name: predict_model(model, a) for name, model in bundle['property_models'].items()}
    if mode == 'ratio':
        if bundle['ratio_model'] is None:
            raise ValueError('Нейросеть Keras ещё не обучена. Выполните полный train.py или ноутбук Google Colab.')
        a = validated_values(bundle['ratio_features'], data)
        return {'Соотношение матрица-наполнитель': predict_model(bundle['ratio_model'], a)}
    raise ValueError('Неизвестный режим расчёта')
