from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

# Тикеры российских компаний
TICKERS = {
    "Сбербанк": "SBER.ME",
    "Газпром": "GAZP.ME",
    "Лукойл": "LKOH.ME",
    "Яндекс": "YNDX.ME",
    "Татнефть": "TATN.ME"
}

def get_stock_keyboard():
    keyboard = [
        [InlineKeyboardButton(text=name, callback_data=ticker)]
        for name, ticker in TICKERS.items()
    ]
    return InlineKeyboardMarkup(inline_keyboard=keyboard)
