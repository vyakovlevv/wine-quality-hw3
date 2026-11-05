import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics


DIR = Path(__file__).resolve().parent
st.set_page_config(page_title='Выбор вина', layout='wide')


@st.cache_resource
def load_model():
    return joblib.load(DIR / 'model.joblib')


@st.cache_data
def load_test():
    return pd.read_csv(DIR / 'test.csv')


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


def predict_table(data, model, features, threshold):
    missing = [col for col in features if col not in data.columns]
    if missing:
        raise ValueError('Не хватает столбцов: ' + ', '.join(missing))
    if data.empty:
        raise ValueError('В файле нет строк с винами.')
    try:
        X = data[features].apply(pd.to_numeric, errors='raise')
    except (TypeError, ValueError):
        raise ValueError('Все признаки должны быть числами.')
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError('В признаках есть пропуски или бесконечные значения.')
    if not X['is_red'].isin([0, 1]).all():
        raise ValueError('В is_red допустимы только 0 (белое) и 1 (красное).')
    result = data.copy()
    result['probability'] = model.predict_proba(X)[:, 1]
    result['prediction'] = (result['probability'] >= threshold).astype(int)
    return result


model = load_model()
test = load_test()
features = list(model.feature_names_in_)
X_test = test[features]
y_test = test['target']
p_test = test_proba(model, X_test)
config = json.loads((DIR / 'config.json').read_text(encoding='utf-8'))

st.title('Выбор вина')
threshold = st.sidebar.slider('Порог', 0.0, 1.0, float(config['threshold']), 0.001, format='%.3f')
st.sidebar.caption('Один порог для всех вкладок.')
one, quality, batch, weights = st.tabs(['Одно вино', 'Модельные метрики', 'CSV', 'Признаки'])

labels = {
    'fixed acidity': 'Фиксированная кислотность',
    'volatile acidity': 'Летучая кислотность',
    'citric acid': 'Лимонная кислота',
    'residual sugar': 'Остаточный сахар',
    'chlorides': 'Хлориды',
    'free sulfur dioxide': 'Свободный диоксид серы',
    'total sulfur dioxide': 'Общий диоксид серы',
    'density': 'Плотность',
    'pH': 'pH',
    'sulphates': 'Сульфаты',
    'alcohol': 'Алкоголь',
}
labels = {name.replace(' ', '_'): label for name, label in labels.items()}

with one:
    values = {}
    cols = st.columns(3)
    for i, col in enumerate(features):
        with cols[i % 3]:
            if col == 'is_red':
                values[col] = int(st.toggle('Красное', value=bool(X_test[col].median())))
            else:
                values[col] = st.number_input(
                    labels.get(col, col), min_value=float(X_test[col].min()),
                    max_value=float(X_test[col].max()), value=float(X_test[col].median()),
                    step=0.0001 if col == 'density' else 0.01,
                    format='%.5f' if col == 'density' else '%.3f')
    p = model.predict_proba(pd.DataFrame([values], columns=features))[0, 1]
    st.metric('Вероятность хорошего вина', f'{p:.1%}')
    st.write('В подборку' if p >= threshold else 'Не в подборку')

with quality:
    pred = (p_test >= threshold).astype(int)
    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred)
    f1 = metrics.f1_score(y_test, pred)
    for col, name, value in zip(st.columns(3), ['Precision', 'Recall', 'F1'], [precision, recall, f1]):
        col.metric(name, f'{value:.3f}')
    cm = metrics.confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics.ConfusionMatrixDisplay(cm, display_labels=['остальные', 'хорошее']).plot(ax=axes[0], colorbar=False)
    metrics.RocCurveDisplay.from_predictions(y_test, p_test, ax=axes[1])
    axes[1].scatter(fp / (fp + tn), recall, color='red', zorder=3)
    axes[1].plot([0, 1], [0, 1], '--', color='gray')
    metrics.PrecisionRecallDisplay.from_predictions(y_test, p_test, ax=axes[2])
    axes[2].scatter(recall, precision if pred.any() else 1.0, color='red', zorder=3)
    axes[2].axhline(y_test.mean(), ls='--', color='gray')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    if not pred.any():
        st.caption('Подборка пуста: precision не определён, в метрике показан 0. На PR-кривой отмечен её край (0, 1).')
    st.write('Выше порог — меньше вин в подборке, обычно меньше посредственных, но больше пропущенных хороших. '
             'Ниже порог — подборка шире, но в неё чаще попадают посредственные вина.')

with batch:
    st.write('Загрузите CSV с теми же признаками, без target.')
    st.download_button('Пример CSV', X_test.head(5).to_csv(index=False).encode('utf-8'),
                       file_name='example.csv', mime='text/csv')
    upload = st.file_uploader('Файл с винами', type='csv')
    if upload is not None:
        try:
            data = pd.read_csv(upload)
            result = predict_table(data, model, features, threshold)
        except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as e:
            st.error(str(e))
        else:
            st.dataframe(result, use_container_width=True)
            st.download_button('Скачать результат', result.to_csv(index=False).encode('utf-8'),
                               file_name='predictions.csv', mime='text/csv')

with weights:
    coefs = pd.Series(model.steps[-1][1].coef_.ravel(), index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    coefs.plot.barh(ax=ax, color=np.where(coefs >= 0, 'seagreen', 'indianred'))
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlabel('Коэффициент после стандартизации')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.write('Сильнее всего за хорошее вино: ' + ', '.join(coefs[coefs > 0].tail(3).index[::-1]) + '.')
    st.write('Сильнее всего против: ' + ', '.join(coefs[coefs < 0].head(3).index) + '.')
