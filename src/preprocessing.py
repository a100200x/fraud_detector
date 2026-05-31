import logging
import os
import pickle
import pandas as pd

logger = logging.getLogger(__name__)

# Путь к сохраненным при обучении артефактам
ARTIFACTS_PATH = './models/preprocessing_artifacts.pkl'


def load_train_data():
    """Вызывается ОДИН РАЗ при старте Docker-контейнера.

    Вместо тяжелого train.csv загружает легкий словарь артефактов из Колаба.
    """
    logger.info('Loading pre-calculated training artifacts for inference...')
    if not os.path.exists(ARTIFACTS_PATH):
        raise FileNotFoundError(
            f'Файл артефактов {ARTIFACTS_PATH} не найден. Пожалуйста, положите его в папку ./models/'
        )

    with open(ARTIFACTS_PATH, 'rb') as f:
        artifacts = pickle.load(f)

    logger.info('Artifacts loaded successfully from %s', ARTIFACTS_PATH)
    return artifacts


def run_preproc(artifacts, input_df):
    """Вызывается для КАЖДОГО нового CSV-файла в папке input.

    Оригинальная функция transform_data, адаптированная под ваш сервис.
    """
    logger.info('Starting preprocessing of incoming dataframe...')
    df = input_df.copy()

    # 1. Удаление неинформативного текста и гео-координат (безопасно, если они уже частично удалены)
    cols_to_drop = [
        'name_1',
        'name_2',
        'street',
        'post_code',
        'lat',
        'lon',
        'merchant_lat',
        'merchant_lon',
    ]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 2. Извлечение временных признаков
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    dt = df['transaction_time'].dt
    df['hour'] = dt.hour
    df['day_of_week'] = dt.dayofweek
    df['day_of_month'] = dt.day
    df.drop(columns=['transaction_time'], inplace=True)

    # 3. Восстановление client_id по исходным данным (берем из input_df, так как в df координаты уже удалены)
    df['client_id'] = (
        input_df['lat'].astype(str)
        + '_'
        + input_df['lon'].astype(str)
        + '_'
        + input_df['gender']
    )

    # 4. Применение поведенческих признаков (отношение к среднему чеку из Колаба)
    df['client_mean_amount'] = (
        df['client_id']
        .map(artifacts['client_mean_amount'])
        .fillna(artifacts['global_mean_amount'])
    )
    df['amount_to_mean_ratio'] = df['amount'] / df['client_mean_amount']
    df.drop(columns=['client_id'], inplace=True)

    # 5. Применение сохраненного сглаженного Target Encoding
    categorical_cols = [
        'gender',
        'merch',
        'cat_id',
        'one_city',
        'us_state',
        'jobs',
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[f'{col}_encoded'] = (
                df[col]
                .map(artifacts['target_encodings'][col])
                .fillna(artifacts['global_target_mean'])
            )

    df.drop(columns=[c for c in categorical_cols if c in df.columns], inplace=True)

    # 6. Гарантируем строгий порядок признаков, на котором обучался CatBoost
    expected_features = [
        'amount',
        'population_city',
        'hour',
        'day_of_week',
        'day_of_month',
        'client_mean_amount',
        'amount_to_mean_ratio',
        'gender_encoded',
        'merch_encoded',
        'cat_id_encoded',
        'one_city_encoded',
        'us_state_encoded',
        'jobs_encoded',
    ]

    # Если вдруг в тесте не хватает колонки, заполняем ее базовым нулем
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]

    logger.info('Preprocessing completed successfully. Shape: %s', df.shape)
    return df
