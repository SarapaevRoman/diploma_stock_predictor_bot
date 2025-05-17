import pandas as pd
import pandas_ta as ta

def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Преобразует исторические данные и добавляет технические индикаторы.

    :param df: DataFrame с колонками ['open','high','low','close','volume','begin',...]
    :return: DataFrame с дополнительными колонками индикаторов
    """
    # Переименовываем и выбираем нужные столбцы
    df = df.rename(columns={
        'begin': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Обеспечиваем корректный тип даты и сортировку
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # Устанавливаем индекс по дате для расчётов индикаторов
    df.set_index('Date', inplace=True)

    # Добавляем простую скользящую среднюю (SMA) за 13 периодов
    df['SMA_13'] = ta.sma(df['Close'], length=13)

    # Добавляем экспоненциальную скользящую среднюю (EMA) за 10 периодов
    df['EMA_10'] = ta.ema(df['Close'], length=10)

    # Добавляем индекс относительной силы (RSI) за 14 периодов
    df['RSI_14'] = ta.rsi(df['Close'], length=14)

    # Добавляем MACD (схождение-расхождение скользящих средних)
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)

    # Добавляем полосы Боллинджера (BBANDS) за 20 периодов и 2 стандартных отклонения
    bb = ta.bbands(df['Close'], length=20, std=2)
    df = pd.concat([df, bb], axis=1)

    # Сброс индекса, чтобы вернуть Date в столбец
    df = df.reset_index()
    return df

def prepare_data_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Подготовка данных для модели. Убираем строки с пропущенными значениями.

    :param df: DataFrame с индикаторами и ценами
    :return: DataFrame с признаками и целевой переменной
    """
    # Убираем строки с NaN значениями, которые могут быть из-за индикаторов
    df = df.dropna()

    # Целевая переменная (например, прогнозируемая цена на следующий день)
    df = df.copy()
    df['Target'] = df['Close'].shift(-1)

    # Убираем строки с NaN значением в целевой переменной
    df = df.dropna(subset=['Target'])

    # Признаки
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_13', 'EMA_10', 'RSI_14', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0']
    X = df[features]
    y = df['Target']
    
    return X, y

