"""Обзорный notebook: реально выполненные ячейки по последнему расчёту train.py."""
import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
cells, scope = [], {}


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})


def code(text):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(text, '<notebook-cell>', 'exec'), scope)
    outputs = [{'output_type': 'stream', 'name': 'stdout', 'text': output.getvalue().splitlines(keepends=True)}]
    cells.append({'cell_type': 'code', 'execution_count': sum(c['cell_type'] == 'code' for c in cells) + 1,
                  'metadata': {}, 'source': text.splitlines(keepends=True), 'outputs': outputs})


def figure(name, title):
    path = ROOT / 'reports' / 'figures' / name
    if not path.exists():
        return
    data = base64.b64encode(path.read_bytes()).decode('ascii')
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': [title + '\n\n', f'![{title}](attachment:{name})'],
                  'attachments': {name: {'image/png': data}}})


md('# Прогнозирование свойств композиционных материалов\n\n'
   'Проект по заданию МГТУ им. Н. Э. Баумана. Python, pandas, NumPy, matplotlib/seaborn, scikit-learn, '
   'MinMaxScaler, Keras/TensorFlow и Flask. Код написан заново.\n\n'
   'Обзорные ячейки выполнены по сохранённым результатам. Полное обучение — `train.py`, журнал — `training_log.txt`. '
   'Состояние нейросети показано явно; pending означает, что обязательный этап ИНС ещё не выполнен.')
md('## 1. Данные и INNER JOIN\n\nПервая колонка Excel — индекс. В результат попадают только общие индексы двух таблиц.')
code("from pathlib import Path\nimport json, sys\nimport pandas as pd\nPROJECT = Path.cwd()\nsys.path.insert(0, str(PROJECT))\nfrom train import load_data, TARGETS, RATIO\nbp, nup, df = load_data()\nprint('Исходные таблицы:', bp.shape, nup.shape, 'После INNER:', df.shape)\nprint('Пропуски:', df.isna().sum().sum(), 'Дубликаты:', df.duplicated().sum())\nprint(df.head(3).to_string())\n")
md('## 2. EDA\n\nВсе гистограммы, boxplot и полная матрица scatter сохранены в `reports/figures`. '
   'Номера переменных расшифрованы в `feature_dictionary.csv`. IQR-выброс не обязательно означает ошибку измерения.')
code("stats = pd.read_csv(PROJECT / 'reports/descriptive_statistics.csv', index_col=0)\nprint(stats[['count', 'mean', 'median', 'min', 'max', 'missing', 'iqr_flags']].to_string())\nprint(df.corr()[TARGETS].to_string())\n")
figure('correlation_heatmap.png', 'Корреляционная матрица')
md('## 3. Разделение и предобработка\n\n716 train и 307 test, random_state=42. '
   'IQR-границы вычисляются только по обучающей части каждого CV-блока. Угол 0/90 не удаляется. '
   'Test и CV-валидация не фильтруются. MinMaxScaler обучается только на очищенной обучающей части.')
code("manifest = json.loads((PROJECT / 'reports/run_manifest.json').read_text(encoding='utf-8'))\nsplit = json.loads((PROJECT / 'reports/split_indices.json').read_text(encoding='utf-8'))\nprint('Train/test:', len(split['train']), len(split['test']))\nprint('Пересечение:', len(set(split['train']) & set(split['test'])))\nprint(pd.read_csv(PROJECT / 'reports/preprocessing_ranges.csv').to_string(index=False))\n")
md('## 4. Работа с моделями\n\nLinearRegression, Ridge, Lasso, RandomForest, GradientBoosting и контрольный средний прогноз. '
   'GridSearchCV с 10 перемешанными блоками; критерий RMSE. '
   'Выбор среди пяти моделей основного варианта с IQR по CV. Дополнительный Ridge без IQR — анализ чувствительности. '
   'Test не использован для выбора. Train-метрики рассчитаны на всех 716 исходных строках, включая исключённые из fit.')
code("metrics = pd.read_csv(PROJECT / 'reports/model_metrics.csv')\nprint(metrics[['target', 'model', 'cv_rmse', 'cv_rmse_std', 'train_r2', 'test_r2', 'test_mae', 'test_rmse']].to_string(index=False))\nprint(json.dumps(manifest['selected_models'], ensure_ascii=False, indent=2))\n")
figure('model_comparison.png', 'R² на независимой тестовой выборке')
md('## 5. ИНС на Keras / TensorFlow\n\nЦель — соотношение матрица–наполнитель. '
   '12 входов включают известные или желаемые механические свойства. Это обратная регрессия, не оптимизация рецепта. '
   'Архитектуры (16,8) и (32,16), ReLU, Adam, линейный выход. '
   'Выбор регуляризации и эпохи по отдельной validation внутри train; test не используется.')
code("print(json.dumps(manifest['neural_network'], ensure_ascii=False, indent=2))\nif manifest['neural_network'].get('status') == 'trained':\n    print(pd.read_csv(PROJECT / 'reports/neural_candidates.csv').to_string(index=False))\n    print(pd.read_csv(PROJECT / 'reports/neural_metrics.csv').to_string(index=False))\nelse:\n    print('Обучение Keras ещё не выполнено. Запустите полный train.py, например в Google Colab.')\n")
figure('neural_learning_curve.png', 'Ошибка обучения и валидации Keras')
md('## 6. Приложение и проверка\n\nИнтерфейс Flask; его зависимости включены в проект. '
   'Прогнозы JSON проверяются против исходных sklearn/Keras-моделей. '
   'При применении не требуется устанавливать TensorFlow. Пример по медианным входам — демонстрация расчёта, не оценка ошибки.')
code("from predictor import load_bundle, predict_request\nbundle = load_bundle(PROJECT / 'models/portable_models.json')\nprint(predict_request(bundle, 'properties', {f['name']: f['default'] for f in bundle['features']}))\nif bundle['ratio_model'] is not None:\n    print(predict_request(bundle, 'ratio', {f['name']: f['default'] for f in bundle['ratio_features']}))\nprint((PROJECT / 'reports/app_verification.json').read_text(encoding='utf-8'))\n")
md('## Выводы\n\n'
   'Сравнивать модели нужно с простым средним и учитывать неопределённость по блокам. '
   'R² около нуля или ниже указывает на слабое качество на данном разделении. '
   'Низкие линейные корреляции не доказывают отсутствия любых нелинейных связей. '
   'Для инженерного использования необходимы уточнение данных и дополнительная валидация.\n\n'
   'Повторное обучение: `python train.py`, затем `python verify.py` и `python make_research_notebook.py`. '
   'Для Google Colab используйте `START_COLAB.ipynb`.')
for i, c in enumerate(cells):
    c['id'] = 'cell-' + str(i + 1)
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                      'language_info': {'name': 'python', 'version': sys.version.split()[0]}},
            'nbformat': 4, 'nbformat_minor': 5}
(ROOT / 'research.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print('Обзорный notebook создан.')
