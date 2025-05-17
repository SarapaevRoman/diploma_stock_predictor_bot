from aiogram import Router, F
from aiogram.types import Message, CallbackQuery, BufferedInputFile
from keyboards import get_stock_keyboard
from stock_utils import load_stock_data
from elt import transform_data, prepare_data_for_model
from predictor import train_model_with_cv, predict_price
from plot_utils import plot_forecast  # Функция возвращает BytesIO
import pandas as pd

router = Router()

@router.message(F.text == "/start")
async def start_command(message: Message):
    # Отправка приветственного сообщения с кнопками выбора акции
    await message.answer("Привет! Выберите акцию для анализа:", reply_markup=get_stock_keyboard())

@router.callback_query()
async def stock_selected(callback: CallbackQuery):
    ticker = callback.data
    await callback.message.edit_text(f"🔄 Загружаю данные по {ticker}...")

    # Загрузка данных
    df = load_stock_data(ticker)
    if df is not None:
        # Преобразование данных
        processed_df = transform_data(df)

        # Подготовка данных для модели
        X, y = prepare_data_for_model(processed_df)

        # Обучение модели
        model, rmse = train_model_with_cv(X, y)

        # Прогнозирование
        predicted_price = predict_price(model, X)  # Передаем модель и данные для прогноза
        predicted_date = processed_df['Date'].iloc[-1] + pd.Timedelta(days=1)

        # Относительная ошибка в процентах
        relative_error_pct = (rmse / predicted_price) * 100

        # Визуализация прогноза
        image_stream = plot_forecast(
            processed_df['Date'],
            processed_df['Close'],
            predicted_date,
            predicted_price
        )

        # Создаём BufferedInputFile из потока
        plot_image_input = BufferedInputFile(
            file=image_stream.getvalue(),
            filename="forecast.png"
        )

        # Отправка картинки
        await callback.message.answer_photo(
            photo=plot_image_input,
            #caption=f"✅ Прогноз завершён.\n🔮 Прогнозируемая цена на {predicted_date.date()}: {predicted_price:.2f}"
             caption=(
                f"✅ Прогноз завершён.\n"
                f"🔮 Прогнозируемая цена на {predicted_date.date()}: {predicted_price:.2f}\n"
                f"📉 Ожидаемая средняя ошибка (RMSE): {rmse:.2f}\n"
                f"📊 Относительная ошибка: {relative_error_pct:.2f}%"
            )
        )

        # Отправка кнопок для выбора акции снова
        await callback.message.answer("Выберите другую акцию для анализа:", reply_markup=get_stock_keyboard())
    else:
        await callback.message.answer("⚠️ Не удалось загрузить данные. Попробуйте позже.")

        # Отправка кнопок для выбора акции снова
        await callback.message.answer("Выберите другую акцию для анализа:", reply_markup=get_stock_keyboard())
