'''
from matplotlib import pyplot as plt
import io

def plot_forecast(dates, prices, predicted_date, predicted_price):
    # Генерируем график
    plt.figure(figsize=(10, 5))
    plt.plot(dates, prices, label="Историческая цена")
    plt.scatter(predicted_date, predicted_price, color='red', label="Прогноз", zorder=5)
    plt.title("Прогноз цены")
    plt.xlabel("Дата")
    plt.ylabel("Цена")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Сохраняем график в память (в байтовый буфер)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return buf  # Возвращаем байтовый поток (BytesIO), а не PIL.Image
'''
from matplotlib import pyplot as plt
import io
import pandas as pd

def plot_forecast(dates, prices, predicted_date, predicted_price):
    # Преобразуем в Series с datetime index для фильтрации последних 60 дней
    data = pd.Series(prices.values, index=pd.to_datetime(dates))
    last_date = data.index.max()
    start_date = last_date - pd.Timedelta(days=60)
    recent_data = data[data.index >= start_date]

    # Генерируем график только за последние 60 дней
    plt.figure(figsize=(10, 5))
    plt.plot(recent_data.index, recent_data.values, label="Историческая цена")
    plt.scatter(predicted_date, predicted_price, color='red', label="Прогноз", zorder=5)
    plt.title("Прогноз цены (последние 2 месяца)")
    plt.xlabel("Дата")
    plt.ylabel("Цена")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return buf
