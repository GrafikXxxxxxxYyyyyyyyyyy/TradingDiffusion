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
            features = []
            
            # 1. Оригинальные цены (уже нормализованы)
            features.append(batch_prices)
            
            # 2. SMA - Simple Moving Averages
            sma_5 = self._sma(batch_prices, 5)
            features.append(sma_5)
            
            sma_10 = self._sma(batch_prices, 10)
            features.append(sma_10)
            
            sma_20 = self._sma(batch_prices, 20)
            features.append(sma_20)
            
            sma_50 = self._sma(batch_prices, 50)
            features.append(sma_50)
            
            # 3. EMA - Exponential Moving Averages
            ema_5 = self._ema(batch_prices, 5)
            features.append(ema_5)
            
            ema_10 = self._ema(batch_prices, 10)
            features.append(ema_10)
            
            ema_20 = self._ema(batch_prices, 20)
            features.append(ema_20)
            
            # 4. RSI
            rsi_14 = self._rsi(batch_prices, 14)
            features.append(rsi_14)
            
            # 5. MACD
            macd_line, signal_line, macd_hist = self._macd(batch_prices)
            features.append(macd_hist)  # Используем гистограмму MACD
            
            # 6. Bollinger Bands
            bb_upper, bb_middle, bb_lower, bb_width = self._bollinger_bands(batch_prices, 20)
            features.append(bb_width)
            
            # 7. Волатильность
            volatility_10 = self._volatility(batch_prices, 10)
            features.append(volatility_10)
            
            volatility_20 = self._volatility(batch_prices, 20)
            features.append(volatility_20)
            
            # 8. Rate of Change
            roc_10 = self._roc(batch_prices, 10)
            features.append(roc_10)
            
            # 9. Williams %R
            williams_r = self._williams_r(batch_prices, 14)
            features.append(williams_r)
            
            # 10. Stochastic Oscillator
            stoch_k, stoch_d = self._stochastic_oscillator(batch_prices, 14)
            features.append(stoch_k)
            
            # 11. Price Rate of Change
            proc_1 = self._price_rate_of_change(batch_prices, 1)
            features.append(proc_1)
            
            # Объединяем все фичи
            batch_features = np.stack(features, axis=1)  # [256, num_features]
            
            # Заменяем все NaN, inf, -inf на 0
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
        return sma


    def _ema(self, prices, window):
        """Exponential Moving Average"""
        alpha = 2 / (window + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema


    def _rsi(self, prices, window=14):
        """Relative Strength Index"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Рассчитываем средние значения для первого периода
        avg_gains = np.zeros_like(prices)
        avg_losses = np.zeros_like(prices)
        
        # Первое значение - простое среднее
        avg_gains[window-1] = np.mean(gains[:window])
        avg_losses[window-1] = np.mean(losses[:window])
        
        # Остальные значения - экспоненциальное сглаживание
        for i in range(window, len(prices)):
            avg_gains[i] = (avg_gains[i-1] * (window-1) + gains[i-1]) / window
            avg_losses[i] = (avg_losses[i-1] * (window-1) + losses[i-1]) / window
        
        # Рассчитываем RSI
        rs = np.divide(avg_gains, avg_losses, out=np.zeros_like(avg_gains), where=avg_losses!=0)
        rsi = 100 - (100 / (1 + rs))
        
        # Заполняем начальные значения
        rsi[:window-1] = 50

        return rsi


    def _macd(self, prices, fast=12, slow=26, signal=9):
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = self._ema(prices, fast)
        ema_slow = self._ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self._ema(macd_line, signal)
        macd_histogram = macd_line - signal_line

        return macd_line, signal_line, macd_histogram


    def _bollinger_bands(self, prices, window=20, num_std=2):
        """Bollinger Bands"""
        sma = self._sma(prices, window)
        std = np.zeros_like(prices)
        
        # Рассчитываем стандартное отклонение
        for i in range(len(prices)):
            start_idx = max(0, i - window + 1)
            std[i] = np.std(prices[start_idx:i+1]) if i >= window-1 else np.std(prices[:i+1])
        
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        band_width = upper_band - lower_band
        
        return upper_band, sma, lower_band, band_width


    def _volatility(self, prices, window):
        """Volatility (Standard Deviation of Returns)"""
        returns = np.diff(prices, prepend=prices[0])
        volatility = np.zeros_like(prices)
        
        for i in range(len(prices)):
            start_idx = max(0, i - window + 1)
            volatility[i] = np.std(returns[start_idx:i+1]) if i >= window-1 else np.std(returns[:i+1])
        
        return volatility


    def _roc(self, prices, window):
        """Rate of Change"""
        roc = np.zeros_like(prices)
        for i in range(window, len(prices)):
            if prices[i-window] != 0:  # Защита от деления на ноль
                roc[i] = (prices[i] - prices[i-window]) / prices[i-window]
            # Если prices[i-window] == 0, остается 0 (по умолчанию)
        return roc


    def _williams_r(self, prices, window=14):
        """Williams %R"""
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
                
        return williams_r


    def _stochastic_oscillator(self, prices, window=14):
        """Stochastic Oscillator"""
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
        
        return k_values, d_values


    def _price_rate_of_change(self, prices, window=1):
        """Price Rate of Change"""
        proc = np.zeros_like(prices)
        for i in range(window, len(prices)):
            if prices[i-window] != 0:
                proc[i] = (prices[i] - prices[i-window]) / prices[i-window]
            else:
                proc[i] = 0
        return proc