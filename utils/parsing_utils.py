import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def parse_snp500(dirpath="data/", 
                 start='2018-01-01', 
                 timeframe='1d', 
                 max_train_len=256, 
                 test_len=32, 
                 split_coeff=0.1):
    """
    Данная функция парсит выборку снп500 если её нет и сохраняет в выбранную директорию 
    dirpath/
        snp500_tickers.csv
        train/
            ticker_1/
                data_0
                data_1
                ...
            ...
        val/
            ticker_1/
                data_0
                data_1
                ...
    """
    # Проверяем, существует ли директория, и создаем её, если нет
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # Если нет таблицы с тикерами всех акций, то её нужно спарсить
    if not os.path.exists(dirpath + 'snp500_tickers.csv'):
        # Получаем pandas таблицу с тикерами и названиями компаний
        sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

        # Сохраняем в csv таблицу матрицу ['Symbol', 'Security']
        sp500_table[['Symbol', 'Security']].to_csv(dirpath + "snp500_tickers.csv", index=False, header=False)

    # Чтение списка тикеров из файла CSV в датафрейм
    tickers_df = pd.read_csv("data/snp500_tickers.csv", header=None, names=['Symbol', 'Security'])

    # Проходимся по всему списку тикеров
    delim = 0
    for index, row in tickers_df.iterrows():
        if delim >= 25:
            return 
            
        ticker = row['Symbol']
        try:
            # Получаем историю при помощи yfinance API
            current_ticker = yf.Ticker(ticker)
            history = current_ticker.history(interval=timeframe, start=start, actions=False, auto_adjust=True, prepost=True)

            # Оставляем только цены закрытия
            close_prices = history[["Close"]]
            
            # Далее в цикле нарезаем с шагом 1 полученные цены на окна заданной длины
            splitted_history = []
            window = (max_train_len + test_len)
            for i in range(0, len(close_prices) - window + 1):
                window_data = close_prices.iloc[i:i+window]
                splitted_history.append(window_data)

            # Берём в качестве теста последние split_coeff окон
            test_items_count = int(len(splitted_history) * split_coeff)
            train, val = splitted_history[:-test_items_count], splitted_history[-test_items_count:]

            # Создаем директории для сохранения, если они еще не существуют
            train_dir = os.path.join(dirpath, 'train', ticker)
            val_dir = os.path.join(dirpath, 'val', ticker)
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)

            # Сохраняем данные train в директории dirpath/train/ticker
            for i, train_df in enumerate(train):
                train_df.to_csv(os.path.join(train_dir, f'sample_{i}.csv'))

            # Сохраняем данные val в директории dirpath/val/ticker
            for i, val_df in enumerate(val):
                val_df.to_csv(os.path.join(val_dir, f'sample_{i}.csv'))
        except:
            print(f"Symbol {ticker} parse failed!")







