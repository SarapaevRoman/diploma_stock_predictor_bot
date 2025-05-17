from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def train_model_with_cv(X: pd.DataFrame, y: pd.Series) -> XGBRegressor:
    """
    Обучает модель XGBoost с кросс-валидацией на подготовленных данных.

    :param X: Признаки
    :param y: Целевая переменная
    :return: Обученная модель XGBoost с лучшими параметрами
    """
    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Инициализация модели
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=5)

    # Параметры для GridSearchCV
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
    }

    # Инициализация GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
    
    # Обучение модели с кросс-валидацией
    grid_search.fit(X_train, y_train)

    # Лучшая модель после кросс-валидации
    best_model = grid_search.best_estimator_

    # Оценка модели на тестовых данных
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')

    return best_model, rmse


def predict_price(model: XGBRegressor, X: pd.DataFrame) -> float:
    """
    Прогнозирует цену на основе обученной модели.

    :param model: Обученная модель XGBoost
    :param X: Признаки для прогноза
    :return: Прогнозируемая цена
    """
    latest_data = X.iloc[-1].values.reshape(1, -1)  # Используем последние данные для прогноза
    predicted_price = model.predict(latest_data)
    return predicted_price[0]
