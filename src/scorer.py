import pandas as pd
import logging
from catboost import CatBoostClassifier
import json
import os
import matplotlib.pyplot as plt


# Настройка логгера
logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model
model = CatBoostClassifier()
model.load_model('./models/my_catboost.cbm')

OUTPUT_DIR = '/app/output'

# Define optimal threshold
model_th = 0.88
logger.info('Pretrained model imported successfully...')


# Make prediction
def make_pred(dt, file_index):
    probabilities = model.predict_proba(dt)[:, 1]

    # Make submission dataframe
    submission = pd.DataFrame({
        'index': file_index,
        'prediction': (probabilities > model_th) * 1
    })
    logger.info('Prediction complete for file: %s', file_index)

    # Дополнительно
    try:
        # 2. Выгрузка ТОП-5 Feature Importances в JSON
        logger.info('Generating top-5 feature importances JSON...')
        importances = model.get_feature_importance()
        feature_names = model.feature_names_

        fi_df = (
            pd.DataFrame({'feature': feature_names, 'importance': importances})
            .sort_values(by='importance', ascending=False)
            .head(5)
        )

        top_5_json = dict(zip(fi_df['feature'], fi_df['importance']))

        json_path = os.path.join(OUTPUT_DIR, 'top_5_features.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(top_5_json, f, indent=4, ensure_ascii=False)
        logger.info('Top-5 features saved to: %s', json_path)

        # 3. Построение графика плотности распределения скоров в PNG
        logger.info('Generating probability density plot...')
        plt.figure(figsize=(10, 6))

        # Строим гистограмму/плотность
        pd.Series(probabilities).plot(
            kind='density', color='crimson', linewidth=2.5
        )

        plt.title('Плотность распределения предсказанных скоров модели')
        plt.xlabel('Вероятность фрода (Score)')
        plt.ylabel('Плотность')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim(0, 1)

        graph_path = os.path.join(OUTPUT_DIR, 'score_density_distribution.png')
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info('Density plot saved to: %s', graph_path)

    except Exception as e:
        logger.error(
            'Ошибка при генерации дополнительных артефактов (JSON/PNG): %s',
            e,
            exc_info=True,
        )
    # Return proba for positive class
    return submission
