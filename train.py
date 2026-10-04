"""Воспроизводимое исследование ВКР. Запуск: python train.py. Все пути от файла."""
import os
# Умеренная нагрузка на ноутбук; фиксируем число потоков до импорта numpy.
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
import copy
import argparse
import json
import platform
import pickle
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from project_model import TrainIQRRegressor
from predictor import predict_model

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / 'reports'
FIGURES = REPORTS / 'figures'
MODELS = ROOT / 'models'
SEED = 42
TARGETS = ['Модуль упругости при растяжении, ГПа', 'Прочность при растяжении, МПа']
RATIO = 'Соотношение матрица-наполнитель'
ANGLE = 'Угол нашивки, град'
SHORT = ['Матрица / наполнитель', 'Плотность', 'Модуль исходного компонента',
         'Отвердитель', 'Эпоксидные группы', 'Температура вспышки',
         'Поверхностная плотность', 'Модуль при растяжении', 'Прочность при растяжении',
         'Потребление смолы', 'Угол нашивки', 'Шаг нашивки', 'Плотность нашивки']


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def load_data():
    bp = pd.read_excel(ROOT / 'data' / 'X_bp.xlsx', index_col=0)
    nup = pd.read_excel(ROOT / 'data' / 'X_nup.xlsx', index_col=0)
    if not bp.index.is_unique or not nup.index.is_unique:
        raise ValueError('Индексы исходных таблиц должны быть уникальными')
    df = bp.join(nup, how='inner', validate='one_to_one').astype(float)
    if not np.isfinite(df.to_numpy()).all():
        raise ValueError('В исходных данных обнаружены нечисловые/пропущенные значения; проверьте файлы')
    return bp, nup, df


def finish_figure(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=160, bbox_inches='tight')
    plt.close()


def eda(df):
    sns.set_theme(style='whitegrid', font='DejaVu Sans')
    stats = df.describe().T
    stats['median'] = df.median()
    stats['missing'] = df.isna().sum()
    q1, q3 = df.quantile(.25), df.quantile(.75)
    stats['iqr_flags'] = ((df < q1 - 1.5 * (q3 - q1)) | (df > q3 + 1.5 * (q3 - q1))).sum()
    stats.to_csv(REPORTS / 'descriptive_statistics.csv', encoding='utf-8-sig')
    df.corr().to_csv(REPORTS / 'correlations.csv', encoding='utf-8-sig')
    for mode in ('hist', 'box'):
        fig, axes = plt.subplots(5, 3, figsize=(13, 16))
        for i, (col, ax) in enumerate(zip(df, axes.flat)):
            if mode == 'hist':
                ax.hist(df[col], bins=25, color='#176C9C', alpha=.9)
                ax.axvline(df[col].mean(), color='#E09315', ls='--', label='Среднее')
                ax.axvline(df[col].median(), color='#B84850', label='Медиана')
            else:
                ax.boxplot(df[col], vert=False, patch_artist=True,
                           boxprops={'facecolor': '#DBEAF3'})
            ax.set_title(SHORT[i], fontsize=10)
        for ax in list(axes.flat)[len(df.columns):]:
            ax.set_visible(False)
        finish_figure('histograms.png' if mode == 'hist' else 'boxplots.png')
    corr = df.rename(columns=dict(zip(df.columns, SHORT))).corr()
    plt.figure(figsize=(13, 11))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1, annot_kws={'fontsize': 8})
    finish_figure('correlation_heatmap.png')
    # Все пары переменных, все 1023 наблюдения. Полная матрица сохранена отдельно.
    fig, axes = plt.subplots(13, 13, figsize=(23, 23))
    for i in range(13):
        for j in range(13):
            ax = axes[i, j]
            if i == j:
                ax.hist(df.iloc[:, i], bins=20, color='#176C9C')
            else:
                ax.scatter(df.iloc[:, j], df.iloc[:, i], s=1, alpha=.18, color='#176C9C', rasterized=True)
            ax.tick_params(labelsize=4)
            if i == 12:
                ax.set_xlabel(str(j + 1), fontsize=8)
            if j == 0:
                ax.set_ylabel(str(i + 1), fontsize=8)
    fig.suptitle('Матрица диаграмм рассеяния. Номера признаков — в feature_dictionary.csv', fontsize=14)
    finish_figure('scatter_matrix.png')


def evaluate(y, prediction):
    return {'r2': float(r2_score(y, prediction)),
            'mae': float(mean_absolute_error(y, prediction)),
            'rmse': float(np.sqrt(mean_squared_error(y, prediction)))}


def make_pipeline(regressor, scale=True):
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scale', MinMaxScaler()))
    steps.append(('model', regressor))
    return Pipeline(steps)


def export_model(fitted):
    p = fitted.estimator_ if isinstance(fitted, TrainIQRRegressor) else fitted
    reg = p.named_steps['model']
    if isinstance(reg, (LinearRegression, Ridge, Lasso)):
        scaler = p.named_steps['scale']
        return {'kind': 'linear', 'factor': scaler.scale_.tolist(), 'offset': scaler.min_.tolist(),
                'coef': reg.coef_.tolist(), 'intercept': float(reg.intercept_)}
    trees = []
    estimators = reg.estimators_.ravel() if isinstance(reg, GradientBoostingRegressor) else reg.estimators_
    for e in estimators:
        t = e.tree_
        trees.append({'left': t.children_left.tolist(), 'right': t.children_right.tolist(),
                      'feature': t.feature.tolist(), 'threshold': t.threshold.tolist(),
                      'value': t.value[:, 0, 0].tolist()})
    if isinstance(reg, GradientBoostingRegressor):
        return {'kind': 'boosting', 'trees': trees, 'learning_rate': reg.learning_rate,
                'init': float(reg.init_.constant_[0, 0])}
    return {'kind': 'forest', 'trees': trees}


def train_properties(df, train_index, test_index):
    features = list(df.drop(columns=TARGETS).columns)
    X_train, X_test = df.loc[train_index, features], df.loc[test_index, features]
    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    angle_index = features.index(ANGLE)
    rows, selected, portable, details = [], {}, {}, {}
    models = [
        ('LinearRegression', make_pipeline(LinearRegression()), {}),
        ('Ridge', make_pipeline(Ridge()), {'estimator__model__alpha': [.001, .01, .1, 1., 10., 100.]}),
        ('Lasso', make_pipeline(Lasso(max_iter=5000)), {'estimator__model__alpha': [.0001, .001, .01, .1, 1.]}),
        ('RandomForest', make_pipeline(RandomForestRegressor(n_estimators=80, random_state=SEED, n_jobs=1), False),
         {'estimator__model__max_depth': [4, None], 'estimator__model__min_samples_leaf': [5, 20]}),
        ('GradientBoosting', make_pipeline(GradientBoostingRegressor(random_state=SEED, learning_rate=.05, min_samples_leaf=10), False),
         {'estimator__model__n_estimators': [50, 100], 'estimator__model__max_depth': [1, 2]})]
    prediction_rows = []
    for target in TARGETS:
        print('\nЦелевая переменная:', target, flush=True)
        y_train, y_test = df.loc[train_index, target], df.loc[test_index, target]
        baseline = make_pipeline(DummyRegressor(strategy='mean'), False)
        scores = -cross_val_score(baseline, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=1)
        baseline.fit(X_train, y_train)
        fitted_models = {'DummyMean': baseline}
        def add_row(name, model, score, score_std, params):
            tr = evaluate(y_train, model.predict(X_train))
            te = evaluate(y_test, model.predict(X_test))
            rows.append({'target': target, 'model': name, 'cv_rmse': float(score), 'cv_rmse_std': float(score_std),
                         **{'train_' + k: v for k, v in tr.items()}, **{'test_' + k: v for k, v in te.items()},
                         'removed_train': getattr(model, 'n_removed_', 0),
                         'parameters': json.dumps(params, ensure_ascii=False)})
            print(name, 'CV RMSE =', round(float(score), 4), 'test R² =', round(te['r2'], 4), flush=True)
        add_row('DummyMean', baseline, scores.mean(), scores.std(), {})
        best_name, best_model, best_score = None, None, float('inf')
        for name, pipe, grid in models:
            candidate = TrainIQRRegressor(pipe, multiplier=1.5, categorical_index=angle_index)
            search = GridSearchCV(candidate, grid, cv=cv, scoring='neg_root_mean_squared_error', n_jobs=1,
                                  return_train_score=True, error_score='raise')
            search.fit(X_train, y_train)
            m = search.best_estimator_
            fitted_models[name] = m
            score = -search.best_score_
            add_row(name, m, score, search.cv_results_['std_test_score'][search.best_index_], search.best_params_)
            slug = 'modulus' if target == TARGETS[0] else 'strength'
            pd.DataFrame(search.cv_results_).to_csv(REPORTS / f'cv_{slug}_{name}.csv', index=False, encoding='utf-8-sig')
            if score < best_score:
                best_name, best_model, best_score = name, m, score
        # Чувствительность к удалению выбросов: те же блоки CV, без выбора по тесту.
        no_iqr = TrainIQRRegressor(make_pipeline(Ridge()), multiplier=None, categorical_index=angle_index)
        search = GridSearchCV(no_iqr, {'estimator__model__alpha': [.001, .01, .1, 1., 10., 100.]},
                              cv=cv, scoring='neg_root_mean_squared_error', n_jobs=1, error_score='raise')
        search.fit(X_train, y_train)
        add_row('RidgeWithoutIQR', search.best_estimator_, -search.best_score_,
                search.cv_results_['std_test_score'][search.best_index_], search.best_params_)
        selected[target] = best_name
        portable[target] = export_model(best_model)
        # Проверка переносимого формата на всех наблюдениях; приложение считает тем же способом.
        expected = best_model.predict(df[features])
        actual = np.array([predict_model(portable[target], row) for row in df[features].to_numpy()])
        err = float(np.max(np.abs(expected - actual)))
        if not np.allclose(expected, actual, rtol=1e-10, atol=1e-8):
            raise AssertionError('JSON-прогноз не совпадает со sklearn')
        slug = 'modulus' if target == TARGETS[0] else 'strength'
        joblib.dump(best_model, MODELS / f'{slug}.joblib')
        with (MODELS / f'model_{slug}.pkl').open('wb') as file:
            pickle.dump(best_model, file)
        baseline_cv = next(x['cv_rmse'] for x in rows if x['target'] == target and x['model'] == 'DummyMean')
        details[target] = {'selected': best_name, 'selection': 'Минимальный CV RMSE среди 5 обучаемых моделей с IQR',
                           'cv_rmse': best_score, 'baseline_cv_rmse': baseline_cv,
                           'beats_baseline_cv': bool(best_score < baseline_cv),
                           'removed_train': best_model.n_removed_, 'portable_max_abs_error': err}
        pred = best_model.predict(X_test)
        for idx, real, forecast in zip(test_index, y_test, pred):
            prediction_rows.append({'index': int(idx), 'target': target, 'actual': float(real), 'prediction': float(forecast)})
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, pred, alpha=.55, s=14, color='#176C9C')
        lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
        plt.plot(lim, lim, '--', color='#E09315', label='Идеальный прогноз')
        plt.xlabel('Фактическое значение'); plt.ylabel('Прогноз')
        plt.title(best_name + ': ' + ('модуль, ГПа' if slug == 'modulus' else 'прочность, МПа'))
        plt.legend()
        finish_figure(f'actual_predicted_{slug}.png')
    metrics = pd.DataFrame(rows)
    metrics.to_csv(REPORTS / 'model_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(prediction_rows).to_csv(REPORTS / 'test_predictions.csv', index=False, encoding='utf-8-sig')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, target in zip(axes, TARGETS):
        subset = metrics[metrics.target == target]
        ax.barh(subset.model, subset.test_r2, color=['#A6ADB5' if m == 'DummyMean' else '#176C9C' for m in subset.model])
        ax.axvline(0, color='#E09315'); ax.set_xlabel('R² на тестовой выборке')
        ax.set_title('Модуль при растяжении' if target == TARGETS[0] else 'Прочность при растяжении')
    finish_figure('model_comparison.png')
    # Границы IQR и параметры масштабирования только по train; категории 0/90 не удаляются.
    ref = TrainIQRRegressor(make_pipeline(Ridge(alpha=100)), categorical_index=angle_index).fit(X_train, df.loc[train_index, TARGETS[0]])
    cleaned = X_train.iloc[ref.keep_mask_]
    scaler = MinMaxScaler().fit(cleaned)
    z = pd.DataFrame(scaler.transform(cleaned), columns=features, index=cleaned.index)
    bounds = pd.DataFrame({'feature': features, 'iqr_lower_train': ref.lower_, 'iqr_upper_train': ref.upper_,
                           'train_min_before': X_train.min().values, 'train_max_before': X_train.max().values,
                           'train_min_clean': cleaned.min().values, 'train_max_clean': cleaned.max().values,
                           'scaled_min': z.min().values, 'scaled_max': z.max().values})
    bounds.to_csv(REPORTS / 'preprocessing_ranges.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame({'index': train_index, 'removed_by_input_iqr': ~ref.keep_mask_}).to_csv(REPORTS / 'outliers_train.csv', index=False, encoding='utf-8-sig')
    fig, axes = plt.subplots(len(features), 2, figsize=(11, 25))
    for i, f in enumerate(features):
        axes[i, 0].hist(cleaned[f], bins=25, color='#176C9C')
        axes[i, 1].hist(z[f], bins=25, color='#E09315')
        axes[i, 0].set_ylabel(SHORT[list(df.columns).index(f)], fontsize=8)
    axes[0, 0].set_title('После IQR, до нормализации'); axes[0, 1].set_title('После MinMaxScaler')
    finish_figure('normalization_before_after.png')
    return features, portable, details


def train_neural(df, train_index, test_index):
    """Обратная регрессия: известные/желаемые свойства -> соотношение. Не оптимизация."""
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    features = list(df.drop(columns=[RATIO]).columns)
    Xtr, Xte = df.loc[train_index, features], df.loc[test_index, features]
    ytr, yte = df.loc[train_index, RATIO], df.loc[test_index, RATIO]
    fit_ids, val_ids = train_test_split(np.arange(len(Xtr)), test_size=.2, random_state=SEED)
    xfit, xval = Xtr.iloc[fit_ids], Xtr.iloc[val_ids]
    yfit, yval = ytr.iloc[fit_ids], ytr.iloc[val_ids]
    # Для ИНС также исключаем выбросы по входам, границы считаем только по fit-подвыборке.
    guard = TrainIQRRegressor(make_pipeline(Ridge()), categorical_index=features.index(ANGLE)).fit(xfit, yfit)
    xfit, yfit = xfit.iloc[guard.keep_mask_], yfit.iloc[guard.keep_mask_]
    xs, ys = MinMaxScaler().fit(xfit), MinMaxScaler().fit(yfit.to_numpy().reshape(-1, 1))
    xf, xv = xs.transform(xfit), xs.transform(xval)
    yf = ys.transform(yfit.to_numpy().reshape(-1, 1)).ravel()
    yv = ys.transform(yval.to_numpy().reshape(-1, 1)).ravel()
    candidates, histories = [], []
    best, best_meta, best_score = None, None, float('inf')
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    for hidden in [(16, 8), (32, 16)]:
        for alpha in [.0001, .001]:
            tf.keras.utils.set_random_seed(SEED)
            model = tf.keras.Sequential([tf.keras.layers.Input(shape=(len(features),))] +
                [tf.keras.layers.Dense(units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(alpha))
                 for units in hidden] + [tf.keras.layers.Dense(1, activation='linear')])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss='mse', metrics=['mse'])
            fit_data = tf.data.Dataset.from_tensor_slices((xf.astype('float32'), yf.astype('float32'))).shuffle(len(yf), seed=SEED).batch(32).with_options(options)
            validation_data = tf.data.Dataset.from_tensor_slices((xv.astype('float32'), yv.astype('float32'))).batch(32).with_options(options)
            stopper = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=20, min_delta=1e-6, restore_best_weights=True)
            history = model.fit(fit_data, validation_data=validation_data, epochs=200, callbacks=[stopper], verbose=0)
            best_epoch = int(stopper.best_epoch + 1)
            tag = str(hidden) + ' alpha=' + str(alpha)
            for epoch, (train_mse, val_mse) in enumerate(zip(history.history['mse'], history.history['val_mse']), 1):
                histories.append({'candidate': tag, 'epoch': epoch,
                                  'train_mse': float(train_mse / ys.scale_[0] ** 2),
                                  'validation_mse': float(val_mse / ys.scale_[0] ** 2)})
            prediction = ys.inverse_transform(model(xv.astype('float32'), training=False).numpy().astype(float)).ravel()
            val_rmse = float(np.sqrt(mean_squared_error(yval, prediction)))
            candidates.append({'hidden_layers': str(hidden), 'alpha': alpha, 'best_epoch': best_epoch,
                               'epochs_run': len(history.history['mse']), 'validation_rmse': val_rmse})
            print('Keras ИНС', tag, 'validation RMSE =', round(val_rmse, 4), 'epoch', best_epoch, flush=True)
            if val_rmse < best_score:
                best, best_score = model, val_rmse
                best_meta = candidates[-1].copy()
    dense_layers = [layer.get_weights() for layer in best.layers]
    portable = {'kind': 'mlp', 'factor': xs.scale_.tolist(), 'offset': xs.min_.tolist(),
                'weights': [w.tolist() for w, b in dense_layers], 'biases': [b.tolist() for w, b in dense_layers],
                'target_offset': float(ys.min_[0]), 'target_factor': float(ys.scale_[0])}
    def forecast(x):
        return ys.inverse_transform(best(xs.transform(x).astype('float32'), training=False).numpy().astype(float)).ravel()
    expected = forecast(df[features])
    actual = np.array([predict_model(portable, r) for r in df[features].to_numpy()])
    if not np.allclose(expected, actual, rtol=1e-6, atol=2e-5):
        raise AssertionError('ИНС JSON не совпадает с Keras')
    baseline_value = float(yfit.mean())
    metrics = []
    for name, predtr, predte in [('KerasMLP', forecast(Xtr), forecast(Xte)),
                                ('DummyMean', np.full(len(Xtr), baseline_value), np.full(len(Xte), baseline_value))]:
        metrics.append({'model': name, **{'train_' + k: v for k, v in evaluate(ytr, predtr).items()},
                        **{'test_' + k: v for k, v in evaluate(yte, predte).items()}})
    pd.DataFrame(metrics).to_csv(REPORTS / 'neural_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(candidates).to_csv(REPORTS / 'neural_candidates.csv', index=False, encoding='utf-8-sig')
    history = pd.DataFrame(histories)
    history.to_csv(REPORTS / 'neural_learning_curves.csv', index=False, encoding='utf-8-sig')
    tag = best_meta['hidden_layers'] + ' alpha=' + str(best_meta['alpha'])
    h = history[history.candidate == tag]
    plt.figure(figsize=(8, 5))
    plt.plot(h.epoch, h.train_mse, label='Обучение'); plt.plot(h.epoch, h.validation_mse, label='Валидация')
    plt.axvline(best_meta['best_epoch'], color='#E09315', ls='--', label='Сохранённая эпоха')
    plt.xlabel('Эпоха'); plt.ylabel('MSE соотношения'); plt.legend()
    finish_figure('neural_learning_curve.png')
    best.save(MODELS / 'ratio_network.keras')
    joblib.dump({'x_scaler': xs, 'y_scaler': ys}, MODELS / 'ratio_scalers.joblib')
    for name, scaler in [('ratio_x_scaler', xs), ('ratio_y_scaler', ys)]:
        with (MODELS / (name + '.pkl')).open('wb') as file:
            pickle.dump(scaler, file)
    details = {**best_meta, 'input_features': len(features), 'activation': 'relu', 'optimizer': 'adam',
               'fit_n_before_iqr': len(fit_ids), 'fit_n_after_iqr': len(yfit), 'validation_n': len(val_ids),
               'test_n': len(test_index), 'early_stopping_patience': 20,
               'framework': 'TensorFlow ' + tf.__version__ + ' / Keras ' + tf.keras.__version__,
               'portable_max_abs_error': float(np.max(np.abs(expected - actual))), 'metrics': metrics}
    return features, portable, details


def feature_metadata(df, columns):
    result = []
    for name in columns:
        i = list(df.columns).index(name)
        f = {'name': name, 'label': name, 'min': float(df[name].min()), 'max': float(df[name].max()),
             'default': float(df[name].median()), 'number': i + 1}
        if name == ANGLE:
            f['choices'] = [0, 90]
        result.append(f)
    return result


def main(skip_neural=False):
    started = time.time()
    for folder in (REPORTS, FIGURES, MODELS):
        folder.mkdir(parents=True, exist_ok=True)
    bp, nup, df = load_data()
    print('INNER JOIN:', df.shape, 'пропуски:', int(df.isna().sum().sum()), 'дубликаты:', int(df.duplicated().sum()), flush=True)
    df.to_csv(REPORTS / 'joined_data.csv', encoding='utf-8-sig')
    pd.DataFrame({'number': range(1, 14), 'column': df.columns, 'short_label': SHORT,
                  'role': ['target' if c in TARGETS else 'input' for c in df.columns]}).to_csv(REPORTS / 'feature_dictionary.csv', index=False, encoding='utf-8-sig')
    train_index, test_index = train_test_split(df.index.to_numpy(), test_size=.3, random_state=SEED)
    save_json(REPORTS / 'split_indices.json', {'random_state': SEED, 'train': train_index.tolist(), 'test': test_index.tolist()})
    eda(df)
    features, property_models, property_details = train_properties(df, train_index, test_index)
    if skip_neural:
        ratio_features = list(df.drop(columns=[RATIO]).columns)
        ratio_model = None
        ratio_details = {'status': 'pending', 'framework': 'Keras / TensorFlow',
                         'reason': 'Требуется выполнение полного train.py, например в Google Colab'}
    else:
        ratio_features, ratio_model, ratio_details = train_neural(df, train_index, test_index)
        ratio_details['status'] = 'trained'
    manifest = {'bp_shape': list(bp.shape), 'nup_shape': list(nup.shape), 'joined_shape': list(df.shape),
                'missing': int(df.isna().sum().sum()), 'duplicates': int(df.duplicated().sum()),
                'train_n': len(train_index), 'test_n': len(test_index), 'random_state': SEED, 'cv_folds': 10,
                'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__,
                'pandas': pd.__version__, 'selected_models': property_details, 'neural_network': ratio_details,
                'duration_seconds': round(time.time() - started, 2)}
    bundle = {'features': feature_metadata(df, features), 'ratio_features': feature_metadata(df, ratio_features),
              'property_models': property_models, 'ratio_model': ratio_model, 'results': manifest}
    save_json(MODELS / 'portable_models.json', bundle)
    save_json(REPORTS / 'run_manifest.json', manifest)
    # Эталонные прогнозы именно исходных моделей, для независимой проверки приложения.
    cases = []
    original_models = {TARGETS[0]: joblib.load(MODELS / 'modulus.joblib'),
                       TARGETS[1]: joblib.load(MODELS / 'strength.joblib')}
    if not skip_neural:
        nn = joblib.load(MODELS / 'ratio_scalers.joblib')
        import tensorflow as tf
        nn_model = tf.keras.models.load_model(MODELS / 'ratio_network.keras')
    for idx in test_index[:5]:
        x = df.loc[[idx], features]
        cases.append({'mode': 'properties', 'values': x.iloc[0].to_dict(),
                      'expected': {name: float(model.predict(x)[0]) for name, model in original_models.items()}})
        if not skip_neural:
            x = df.loc[[idx], ratio_features]
            prediction = nn['y_scaler'].inverse_transform(
                nn_model(nn['x_scaler'].transform(x).astype('float32'), training=False).numpy().astype(float))[0, 0]
            cases.append({'mode': 'ratio', 'values': x.iloc[0].to_dict(), 'expected': {RATIO: float(prediction)}})
    save_json(REPORTS / 'verification_cases.json', cases)
    print('\nГотово. Время, секунд:', manifest['duration_seconds'], flush=True)
    print('Модели:', MODELS, '\nРезультаты:', REPORTS, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-neural', action='store_true', help='Только регрессионные модели; состояние ИНС отмечается pending')
    main(skip_neural=parser.parse_args().skip_neural)
