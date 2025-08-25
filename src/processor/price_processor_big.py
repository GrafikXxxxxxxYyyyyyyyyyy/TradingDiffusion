import torch
import numpy as np


class TradingGDTProcessor:
    def __init__(self):
        pass


    def __call__(self, close_prices):
        """
        Предобрабатывает историю цен закрытия, добавляя к ним осцилляторы в качестве новых фичей
        
        Args:
            close_prices: torch.Tensor of shape [batch_size, 256, 1] 
            
        Returns:
            torch.Tensor of shape [batch_size, 256, num_features]
        """
        # Проверка на NaN и Inf в входных данных
        if torch.isnan(close_prices).any() or torch.isinf(close_prices).any():
            close_prices = torch.nan_to_num(close_prices, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Убираем последнее измерение, чтобы получить [batch_size, 256]
        prices_squeezed = close_prices.squeeze(-1)  # [batch_size, 256]
        
        batch_size, seq_len = prices_squeezed.shape
        device = prices_squeezed.device
        
        # Конвертируем в numpy для удобства вычислений
        prices_np = prices_squeezed.cpu().numpy()
        
        # Собираем все фичи
        all_features = []
        
        for i in range(batch_size):
            batch_prices = prices_np[i]  # [256]
            
            # Проверка на NaN в батче
            if np.isnan(batch_prices).any() or np.isinf(batch_prices).any():
                batch_prices = np.nan_to_num(batch_prices, nan=0.0, posinf=0.0, neginf=0.0)
            
            features = []
            
            # 1. Оригинальные цены (уже нормализованы)
            features.append(batch_prices)
            
            # 2. SMA - Simple Moving Averages
            sma_5 = self._sma(batch_prices, 5)
            sma_5 = np.nan_to_num(sma_5, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(sma_5)
            
            sma_10 = self._sma(batch_prices, 10)
            sma_10 = np.nan_to_num(sma_10, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(sma_10)
            
            sma_20 = self._sma(batch_prices, 20)
            sma_20 = np.nan_to_num(sma_20, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(sma_20)
            
            sma_50 = self._sma(batch_prices, 50)
            sma_50 = np.nan_to_num(sma_50, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(sma_50)
            
            # 3. EMA - Exponential Moving Averages
            ema_5 = self._ema(batch_prices, 5)
            ema_5 = np.nan_to_num(ema_5, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(ema_5)
            
            ema_10 = self._ema(batch_prices, 10)
            ema_10 = np.nan_to_num(ema_10, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(ema_10)
            
            ema_20 = self._ema(batch_prices, 20)
            ema_20 = np.nan_to_num(ema_20, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(ema_20)
            
            # 4. RSI
            rsi_14 = self._rsi(batch_prices, 14)
            rsi_14 = np.nan_to_num(rsi_14, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(rsi_14)
            
            # 5. MACD
            macd_line, signal_line, macd_hist = self._macd(batch_prices)
            macd_hist = np.nan_to_num(macd_hist, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(macd_hist)  # Используем гистограмму MACD
            
            # 6. Bollinger Bands
            bb_upper, bb_middle, bb_lower, bb_width = self._bollinger_bands(batch_prices, 20)
            bb_width = np.nan_to_num(bb_width, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(bb_width)
            
            # 7. Волатильность
            volatility_10 = self._volatility(batch_prices, 10)
            volatility_10 = np.nan_to_num(volatility_10, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(volatility_10)
            
            volatility_20 = self._volatility(batch_prices, 20)
            volatility_20 = np.nan_to_num(volatility_20, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(volatility_20)
            
            # 8. Rate of Change
            roc_10 = self._roc(batch_prices, 10)
            roc_10 = np.nan_to_num(roc_10, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(roc_10)
            
            # 9. Williams %R
            williams_r = self._williams_r(batch_prices, 14)
            williams_r = np.nan_to_num(williams_r, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(williams_r)
            
            # 10. Stochastic Oscillator
            stoch_k, stoch_d = self._stochastic_oscillator(batch_prices, 14)
            stoch_k = np.nan_to_num(stoch_k, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(stoch_k)
            
            # 11. Price Rate of Change
            proc_1 = self._price_rate_of_change(batch_prices, 1)
            proc_1 = np.nan_to_num(proc_1, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(proc_1)
            
            # 12. ATR - Average True Range
            atr_14 = self._atr(batch_prices, batch_prices, batch_prices, 14)  # Для простоты используем close для всех
            atr_14 = np.nan_to_num(atr_14, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(atr_14)
            
            # 13. CCI - Commodity Channel Index
            cci_20 = self._cci(batch_prices, batch_prices, batch_prices, 20)
            cci_20 = np.nan_to_num(cci_20, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(cci_20)
            
            # 14. OBV - On Balance Volume (упрощенная версия)
            obv = self._obv(batch_prices, np.ones_like(batch_prices))  # Используем единицы как объем
            obv = np.nan_to_num(obv, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(obv)
            
            # 15. ADX - Average Directional Index
            adx_14 = self._adx(batch_prices, batch_prices, batch_prices, 14)
            adx_14 = np.nan_to_num(adx_14, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(adx_14)
            
            # 16. Momentum
            momentum_10 = self._momentum(batch_prices, 10)
            momentum_10 = np.nan_to_num(momentum_10, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(momentum_10)
            
            # 17. Chaikin Oscillator
            chaikin_3_10 = self._chaikin_oscillator(batch_prices, batch_prices, batch_prices, np.ones_like(batch_prices), 3, 10)
            chaikin_3_10 = np.nan_to_num(chaikin_3_10, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(chaikin_3_10)
            
            # 18. MFI - Money Flow Index
            mfi_14 = self._mfi(batch_prices, batch_prices, batch_prices, np.ones_like(batch_prices), 14)
            mfi_14 = np.nan_to_num(mfi_14, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(mfi_14)
            
            # 19. Stochastic RSI
            stoch_rsi = self._stochastic_rsi(batch_prices, 14)
            stoch_rsi = np.nan_to_num(stoch_rsi, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(stoch_rsi)
            
            # 20. Ultimate Oscillator
            ultimate_osc = self._ultimate_oscillator(batch_prices, batch_prices, batch_prices, 7, 14, 28)
            ultimate_osc = np.nan_to_num(ultimate_osc, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(ultimate_osc)
            
            # 21. TRIX
            trix_15 = self._trix(batch_prices, 15)
            trix_15 = np.nan_to_num(trix_15, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(trix_15)
            
            # 22. Mass Index
            mass_index = self._mass_index(batch_prices, batch_prices, 9, 25)
            mass_index = np.nan_to_num(mass_index, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(mass_index)
            
            # 23. Vortex Indicator
            vortex_plus, vortex_minus = self._vortex_indicator(batch_prices, batch_prices, batch_prices, 14)
            vortex_plus = np.nan_to_num(vortex_plus, nan=0.0, posinf=0.0, neginf=0.0)
            vortex_minus = np.nan_to_num(vortex_minus, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(vortex_plus)
            
            # 24. KST - Know Sure Thing
            kst = self._kst(batch_prices)
            kst = np.nan_to_num(kst, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(kst)
            
            # 25. Elder Ray Index
            bull_power, bear_power = self._elder_ray(batch_prices, 13)
            bull_power = np.nan_to_num(bull_power, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(bull_power)
            
            # 26. Fisher Transform
            fisher_transform = self._fisher_transform(batch_prices, 10)
            fisher_transform = np.nan_to_num(fisher_transform, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(fisher_transform)
            
            # Объединяем все фичи
            batch_features = np.stack(features, axis=1)  # [256, num_features]
            
            # Заменяем все NaN, inf, -inf на 0 (дополнительная проверка)
            batch_features = np.nan_to_num(batch_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            all_features.append(batch_features)
        
        # Конвертируем обратно в torch tensor
        result = torch.FloatTensor(np.array(all_features)).to(device)  # [batch_size, 256, num_features]
        
        # Финальная замена NaN на 0 (на всякий случай)
        result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            
        return result


    def _sma(self, prices, window):
        """Simple Moving Average"""
        padded_prices = np.pad(prices, (window-1, 0), mode='edge')
        cumsum = np.cumsum(padded_prices)
        sma = (cumsum[window-1:] - np.concatenate([[0], cumsum[:-window]])) / window
        return np.nan_to_num(sma, nan=0.0, posinf=0.0, neginf=0.0)


    def _ema(self, prices, window):
        """Exponential Moving Average"""
        alpha = 2 / (window + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0] if not (np.isnan(prices[0]) or np.isinf(prices[0])) else 0
        for i in range(1, len(prices)):
            if np.isnan(prices[i]) or np.isinf(prices[i]):
                prices[i] = 0
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return np.nan_to_num(ema, nan=0.0, posinf=0.0, neginf=0.0)


    def _rsi(self, prices, window=14):
        """Relative Strength Index"""
        # Проверка на NaN в ценах
        prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Рассчитываем средние значения для первого периода
        avg_gains = np.zeros_like(prices)
        avg_losses = np.zeros_like(prices)
        
        # Первое значение - простое среднее
        avg_gains[window-1] = np.mean(gains[:window]) if len(gains[:window]) > 0 else 0
        avg_losses[window-1] = np.mean(losses[:window]) if len(losses[:window]) > 0 else 0
        
        # Остальные значения - экспоненциальное сглаживание
        for i in range(window, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (window-1) + gains[i-1]) / window
            avg_losses[i] = (avg_losses[i-1] * (window-1) + losses[i-1]) / window
        
        # Рассчитываем RSI
        rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses!=0)
        rsi = 100 - (100 / (1 + rs))
        
        # Заполняем начальные значения
        rsi[:window-1] = 50

        return np.nan_to_num(rsi, nan=0.0, posinf=0.0, neginf=0.0)


    def _macd(self, prices, fast=12, slow=26, signal=9):
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        macd_histogram = macd_line - signal_line

        return np.nan_to_num(macd_line, nan=0.0, posinf=0.0, neginf=0.0), \
               np.nan_to_num(signal_line, nan=0.0, posinf=0.0, neginf=0.0), \
               np.nan_to_num(macd_histogram, nan=0.0, posinf=0.0, neginf=0.0)


    def _bollinger_bands(self, prices, window=20, num_std=2):
        """Bollinger Bands"""
        # Проверка на NaN
        prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        
        sma = self._sma(prices, window)
        std = np.zeros_like(prices)
        
        # Рассчитываем стандартное отклонение
        for i in range(len(prices)):
            start_idx = max(0, i - window + 1)
            std[i] = np.std(prices[start_idx:i+1]) if i >= window-1 and len(prices[start_idx:i+1]) > 0 else 0
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        band_width = upper_band - lower_band
        
        return np.nan_to_num(upper_band, nan=0.0, posinf=0.0, neginf=0.0), \
               np.nan_to_num(sma, nan=0.0, posinf=0.0, neginf=0.0), \
               np.nan_to_num(lower_band, nan=0.0, posinf=0.0, neginf=0.0), \
               np.nan_to_num(band_width, nan=0.0, posinf=0.0, neginf=0.0)


    def _volatility(self, prices, window):
        """Volatility (Standard Deviation of Returns)"""
        # Проверка на NaN
        prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        
        returns = np.diff(prices, prepend=prices[0])
        volatility = np.zeros_like(prices)
        
        for i in range(len(prices)):
            start_idx = max(0, i - window + 1)
            if i >= window-1 and len(returns[start_idx:i+1]) > 0:
                volatility[i] = np.std(returns[start_idx:i+1])
            elif len(returns[:i+1]) > 0:
                volatility[i] = np.std(returns[:i+1])
        
        return np.nan_to_num(volatility, nan=0.0, posinf=0.0, neginf=0.0)


    def _roc(self, prices, window):
        """Rate of Change"""
        # Проверка на NaN
        prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        
        roc = np.zeros_like(prices)
        for i in range(window, len(prices)):
            if prices[i-window] != 0:  # Защита от деления на ноль
                roc[i] = (prices[i] - prices[i-window]) / prices[i-window]
            # Если prices[i-window] == 0, остается 0 (по умолчанию)
        return np.nan_to_num(roc, nan=0.0, posinf=0.0, neginf=0.0)


    def _williams_r(self, prices, window=14):
        """Williams %R"""
        # Проверка на NaN
        prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        
        williams_r = np.zeros_like(prices)
        
        for i in range(len(prices)):
            start_idx = max(0, i - window + 1)
            period_prices = prices[start_idx:i+1]
            
            if len(period_prices) >= window:
                highest_high = np.max(period_prices)
                lowest_low = np.min(period_prices)
                
                if highest_high != lowest_low:
                    williams_r[i] = (highest_high - prices[i]) / (highest_high - lowest_low) * -100
                else:
                    williams_r[i] = -50
            else:
                williams_r[i] = -50
                
        return np.nan_to_num(williams_r, nan=0.0, posinf=0.0, neginf=0.0)


    def _stochastic_oscillator(self, prices, window=14):
        """Stochastic Oscillator"""
        # Проверка на NaN
        prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        
        k_values = np.zeros_like(prices)
        d_values = np.zeros_like(prices)
        
        for i in range(len(prices)):
            start_idx = max(0, i - window + 1)
            period_prices = prices[start_idx:i+1]
            
            if len(period_prices) >= window:
                highest_high = np.max(period_prices)
                lowest_low = np.min(period_prices)
                
                if highest_high != lowest_low:
                    k_values[i] = (prices[i] - lowest_low) / (highest_high - lowest_low) * 100
                else:
                    k_values[i] = 50
            else:
                k_values[i] = 50
        
        # Сглаживаем для получения %D линии
        d_values = self._sma(k_values, 3)
        
        return np.nan_to_num(k_values, nan=0.0, posinf=0.0, neginf=0.0), \
               np.nan_to_num(d_values, nan=0.0, posinf=0.0, neginf=0.0)


    def _price_rate_of_change(self, prices, window=1):
        """Price Rate of Change"""
        # Проверка на NaN
        prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        
        proc = np.zeros_like(prices)
        for i in range(window, len(prices)):
            if prices[i-window] != 0:
                proc[i] = (prices[i] - prices[i-window]) / prices[i-window]
            else:
                proc[i] = 0
        return np.nan_to_num(proc, nan=0.0, posinf=0.0, neginf=0.0)


    # Новые осцилляторы
    
    def _atr(self, high, low, close, window):
        """Average True Range"""
        # Проверка на NaN
        high = np.nan_to_num(high, nan=0.0, posinf=0.0, neginf=0.0)
        low = np.nan_to_num(low, nan=0.0, posinf=0.0, neginf=0.0)
        close = np.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)
        
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        atr = np.zeros_like(tr)
        atr[0] = tr[0] if not (np.isnan(tr[0]) or np.isinf(tr[0])) else 0
        for i in range(1, len(tr)):
            if np.isnan(tr[i]) or np.isinf(tr[i]):
                tr[i] = 0
            atr[i] = (atr[i-1] * (window-1) + tr[i]) / window
        return np.nan_to_num(atr, nan=0.0, posinf=0.0, neginf=0.0)


    def _cci(self, high, low, close, window):
        """Commodity Channel Index"""
        # Проверка на NaN
        high = np.nan_to_num(high, nan=0.0, posinf=0.0, neginf=0.0)
        low = np.nan_to_num(low, nan=0.0, posinf=0.0, neginf=0.0)
        close = np.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)
        
        tp = (high + low + close) / 3
        sma_tp = self._sma(tp, window)
        mean_deviation = np.zeros_like(tp)
        
        for i in range(len(tp)):
            start_idx = max(0, i - window + 1)
            if i >= window-1 and len(tp[start_idx:i+1]) > 0:
                mean_deviation[i] = np.mean(np.abs(tp[start_idx:i+1] - sma_tp[i]))
            elif len(tp[:i+1]) > 0:
                mean_deviation[i] = np.mean(np.abs(tp[:i+1] - sma_tp[i]))
        
        cci = np.divide(tp - sma_tp, 0.015 * mean_deviation, out=np.zeros_like(tp), where=mean_deviation!=0)
        return np.nan_to_num(cci, nan=0.0, posinf=0.0, neginf=0.0)


    def _obv(self, close, volume):
        """On Balance Volume"""
        # Проверка на NaN
        close = np.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
        
        obv = np.zeros_like(close)
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        return np.nan_to_num(obv, nan=0.0, posinf=0.0, neginf=0.0)


    def _adx(self, high, low, close, window):
        """Average Directional Index"""
        # Проверка на NaN
        high = np.nan_to_num(high, nan=0.0, posinf=0.0, neginf=0.0)
        low = np.nan_to_num(low, nan=0.0, posinf=0.0, neginf=0.0)
        close = np.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Рассчитываем +DI и -DI
        up_move = np.diff(high, prepend=high[0])
        down_move = np.diff(low, prepend=low[0]) * -1
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        atr = self._atr(high, low, close, window)
        
        plus_di = np.divide(plus_dm, atr, out=np.zeros_like(plus_dm), where=atr!=0) * 100
        minus_di = np.divide(minus_dm, atr, out=np.zeros_like(minus_dm), where=atr!=0) * 100
        
        # Рассчитываем DX
        dx = np.divide(np.abs(plus_di - minus_di), plus_di + minus_di, out=np.zeros_like(plus_di), where=(plus_di + minus_di)!=0) * 100
        
        # Рассчитываем ADX
        adx = np.zeros_like(dx)
        adx[window-1] = np.mean(dx[:window]) if len(dx[:window]) > 0 else 0
        for i in range(window, len(dx)):
            adx[i] = (adx[i-1] * (window-1) + dx[i]) / window
            
        return np.nan_to_num(adx, nan=0.0, posinf=0.0, neginf=0.0)


    def _momentum(self, prices, window):
        """Momentum"""
        # Проверка на NaN
        prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        
        momentum = np.zeros_like(prices)
        for i in range(window, len(prices)):
            momentum[i] = prices[i] - prices[i-window]
        return np.nan_to_num(momentum, nan=0.0, posinf=0.0, neginf=0.0)


    def _chaikin_oscillator(self, high, low, close, volume, fast=3, slow=10):
        """Chaikin Oscillator"""
        # Проверка на NaN
        high = np.nan_to_num(high, nan=0.0, posinf=0.0, neginf=0.0)
        low = np.nan_to_num(low, nan=0.0, posinf=0.0, neginf=0.0)
        close = np.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Рассчитываем Accumulation/Distribution Line с защитой от деления на ноль
        high_low_diff = high - low
        clv = np.divide((close - low) - (high - close), high_low_diff, out=np.zeros_like(close), where=high_low_diff!=0)
        adl = np.cumsum(clv * volume)
        
        # Рассчитываем Chaikin Oscillator
        ema_fast = self._ema(adl, fast)
        ema_slow = self._ema(adl, slow)
        chaikin = ema_fast - ema_slow
        
        return np.nan_to_num(chaikin, nan=0.0, posinf=0.0, neginf=0.0)


    def _mfi(self, high, low, close, volume, window):
        """Money Flow Index"""
        # Проверка на NaN
        high = np.nan_to_num(high, nan=0.0, posinf=0.0, neginf=0.0)
        low = np.nan_to_num(low, nan=0.0, posinf=0.0, neginf=0.0)
        close = np.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)
        volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
        
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        # Positive and negative money flow
        positive_flow = np.zeros_like(money_flow)
        negative_flow = np.zeros_like(money_flow)
        
        for i in range(1, len(typical_price)):
            if typical_price[i] > typical_price[i-1]:
                positive_flow[i] = money_flow[i]
            elif typical_price[i] < typical_price[i-1]:
                negative_flow[i] = money_flow[i]
        
        # Рассчитываем MFI с защитой от деления на ноль
        positive_mf = self._sma(positive_flow, window)
        negative_mf = self._sma(negative_flow, window)
        
        total_mf = positive_mf + negative_mf
        mfi = np.divide(positive_mf, total_mf, out=np.zeros_like(positive_mf), where=total_mf!=0) * 100
        
        return np.nan_to_num(mfi, nan=0.0, posinf=0.0, neginf=0.0)


    def _stochastic_rsi(self, prices, window):
        """Stochastic RSI"""
        # Проверка на NaN
        prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        
        rsi = self._rsi(prices, window)
        stoch_rsi = np.zeros_like(rsi)
        
        for i in range(len(rsi)):
            start_idx = max(0, i - window + 1)
            period_rsi = rsi[start_idx:i+1]
            
            if len(period_rsi) >= window:
                min_rsi = np.min(period_rsi)
                max_rsi = np.max(period_rsi)
                if max_rsi != min_rsi:
                    stoch_rsi[i] = (rsi[i] - min_rsi) / (max_rsi - min_rsi) * 100
                else:
                    stoch_rsi[i] = 50
            else:
                stoch_rsi[i] = 50
                
        return np.nan_to_num(stoch_rsi, nan=0.0, posinf=0.0, neginf=0.0)


    def _ultimate_oscillator(self, high, low, close, short=7, medium=14, long=28):
        """Ultimate Oscillator"""
        # Проверка на NaN
        high = np.nan_to_num(high, nan=0.0, posinf=0.0, neginf=0.0)
        low = np.nan_to_num(low, nan=0.0, posinf=0.0, neginf=0.0)
        close = np.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)
        
        bp = close - np.minimum(low, np.roll(close, 1))
        bp[0] = close[0] - low[0] if len(low) > 0 else 0
        tr = np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))
        tr[0] = high[0] - low[0] if len(high) > 0 and len(low) > 0 else 0
        
        # Рассчитываем average для каждого периода с защитой от деления на ноль
        def calc_average(period):
            avg = np.zeros_like(bp)
            for i in range(period-1, len(bp)):
                tr_sum = np.sum(tr[i-period+1:i+1])
                bp_sum = np.sum(bp[i-period+1:i+1])
                if tr_sum != 0:
                    avg[i] = bp_sum / tr_sum
            return np.nan_to_num(avg, nan=0.0, posinf=0.0, neginf=0.0)
        
        avg_short = calc_average(short)
        avg_medium = calc_average(medium)
        avg_long = calc_average(long)
        
        # Защита от деления на ноль в финальном расчете
        denominator = (4 + 2 + 1)  # 7
        if denominator != 0:
            ultimate = 100 * ((4 * avg_short) + (2 * avg_medium) + avg_long) / denominator
        else:
            ultimate = np.zeros_like(avg_short)
        
        return np.nan_to_num(ultimate, nan=0.0, posinf=0.0, neginf=0.0)


    def _trix(self, prices, window):
        """TRIX"""
        # Проверка на NaN
        prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        
        ema1 = self._ema(prices, window)
        ema2 = self._ema(ema1, window)
        ema3 = self._ema(ema2, window)
        
        trix = np.zeros_like(ema3)
        for i in range(1, len(ema3)):
            if ema3[i-1] != 0:
                trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
        return np.nan_to_num(trix, nan=0.0, posinf=0.0, neginf=0.0)


    def _mass_index(self, high, low, ema_period=9, sum_period=25):
        """Mass Index"""
        # Проверка на NaN
        high = np.nan_to_num(high, nan=0.0, posinf=0.0, neginf=0.0)
        low = np.nan_to_num(low, nan=0.0, posinf=0.0, neginf=0.0)
        
        ema = self._ema(high - low, ema_period)
        ema_ema = self._ema(ema, ema_period)
        
        # Защита от деления на ноль
        ratio = np.divide(ema, ema_ema, out=np.zeros_like(ema), where=ema_ema!=0)
        
        mass = np.zeros_like(ratio)
        for i in range(sum_period-1, len(ratio)):
            mass[i] = np.sum(ratio[i-sum_period+1:i+1])
        return np.nan_to_num(mass, nan=0.0, posinf=0.0, neginf=0.0)


    def _vortex_indicator(self, high, low, close, window):
        """Vortex Indicator"""
        # Проверка на NaN
        high = np.nan_to_num(high, nan=0.0, posinf=0.0, neginf=0.0)
        low = np.nan_to_num(low, nan=0.0, posinf=0.0, neginf=0.0)
        close = np.nan_to_num(close, nan=0.0, posinf=0.0, neginf=0.0)
        
        tr = np.zeros_like(high)
        plus_vm = np.zeros_like(high)
        minus_vm = np.zeros_like(high)
        
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            plus_vm[i] = abs(high[i] - low[i-1])
            minus_vm[i] = abs(low[i] - high[i-1])
        
        # Вычисляем SMA
        sma_plus_vm = self._sma(plus_vm, window)
        sma_minus_vm = self._sma(minus_vm, window)
        sma_tr = self._sma(tr, window)
        
        # Защита от деления на ноль
        plus_vi = np.divide(sma_plus_vm, sma_tr, out=np.zeros_like(sma_tr), where=sma_tr!=0)
        minus_vi = np.divide(sma_minus_vm, sma_tr, out=np.zeros_like(sma_tr), where=sma_tr!=0)
        
        return np.nan_to_num(plus_vi, nan=0.0, posinf=0.0, neginf=0.0), \
               np.nan_to_num(minus_vi, nan=0.0, posinf=0.0, neginf=0.0)


    def _kst(self, prices):
        """Know Sure Thing"""
        # Проверка на NaN
        prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        
        roc1 = self._roc(prices, 10)
        roc2 = self._roc(prices, 15)
        roc3 = self._roc(prices, 20)
        roc4 = self._roc(prices, 30)
        
        kst = self._sma(roc1, 10) + self._sma(roc2, 10) * 2 + self._sma(roc3, 10) * 3 + self._sma(roc4, 15) * 4
        return np.nan_to_num(kst, nan=0.0, posinf=0.0, neginf=0.0)


    def _elder_ray(self, prices, window):
        """Elder Ray Index"""
        # Проверка на NaN
        prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        
        ema = self._ema(prices, window)
        bull_power = prices - ema
        bear_power = prices - ema  # В упрощенной версии
        return np.nan_to_num(bull_power, nan=0.0, posinf=0.0, neginf=0.0), \
               np.nan_to_num(bear_power, nan=0.0, posinf=0.0, neginf=0.0)


    def _fisher_transform(self, prices, window):
        """Fisher Transform"""
        # Проверка на NaN
        prices = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Нормализуем цены в диапазон [-1, 1]
        if len(prices) >= window:
            min_price = np.minimum.accumulate(prices)[-window:]
            max_price = np.maximum.accumulate(prices)[-window:]
        else:
            min_price = prices
            max_price = prices
        
        if len(min_price) > 0 and len(max_price) > 0:
            price_range = max_price[-1] - min_price[-1] + 1e-8
            if price_range != 0:
                normalized = ((prices - min_price[-1]) / price_range) * 2 - 1
                normalized = np.clip(normalized, -0.9999, 0.9999)
                # Защита от логарифма отрицательных значений
                denominator = 1 - normalized
                numerator = 1 + normalized
                # Защита от деления на ноль и логарифма <= 0
                valid_mask = (denominator > 1e-10) & (numerator > 1e-10)
                fisher = np.zeros_like(normalized)
                fisher[valid_mask] = 0.5 * np.log(numerator[valid_mask] / denominator[valid_mask])
            else:
                fisher = np.zeros_like(prices)
        else:
            fisher = np.zeros_like(prices)
            
        return np.nan_to_num(fisher, nan=0.0, posinf=0.0, neginf=0.0)