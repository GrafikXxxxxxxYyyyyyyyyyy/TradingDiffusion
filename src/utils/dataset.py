import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import glob


class TradingGDTDataset(Dataset):
    def __init__(self, data_path, mode='train', transform=None):
        """
        Custom dataset for trading data
        
        Args:
            data_path (str): path to data directory
            mode (str): 'train' or 'validation'
            transform (callable, optional): optional transform to be applied on a sample
        """
        self.data_path = data_path
        self.mode = mode
        self.transform = transform
        
        # Проверяем существование директории
        mode_path = os.path.join(data_path, mode)
        if not os.path.exists(mode_path):
            raise FileNotFoundError(f"Mode directory not found: {mode_path}")
        
        # Получаем все пути к тикерам
        self.ticker_paths = glob.glob(os.path.join(mode_path, '*'))
        
        if not self.ticker_paths:
            raise ValueError(f"No ticker directories found in {mode_path}")
        
        # Собираем все пары history-target файлов
        self.samples = []
        self._collect_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid history-target pairs found in {mode_path}")
        
        print(f"Found {len(self.samples)} samples for {mode} mode")
    
    
    def _collect_samples(self):
        """Собирает все доступные пары history-target файлов"""
        for ticker_path in self.ticker_paths:
            if not os.path.isdir(ticker_path):
                continue
                
            # Получаем все history файлы для данного тикера
            history_files = glob.glob(os.path.join(ticker_path, 'history_*.csv'))
            
            for history_file in history_files:
                # Проверяем, что history файл не пустой
                if not os.path.exists(history_file) or os.path.getsize(history_file) == 0:
                    continue
                    
                # Получаем индекс файла
                file_basename = os.path.basename(history_file)
                if file_basename.startswith('history_'):
                    file_index = file_basename.replace('history_', '').replace('.csv', '')
                    target_file = os.path.join(ticker_path, f'target_{file_index}.csv')
                    
                    # Проверяем существование и непустоту соответствующего target файла
                    if os.path.exists(target_file) and os.path.getsize(target_file) > 0:
                        self.samples.append({
                            'history_file': history_file,
                            'target_file': target_file,
                            'ticker': os.path.basename(ticker_path)
                        })
    
    
    def __len__(self):
        """Возвращает общее количество samples"""
        return len(self.samples)
    

    def __getitem__(self, idx):
        """Возвращает один sample по индексу"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample_info = self.samples[idx]
        
        try:
            # Загружаем history данные с проверкой
            history_df = pd.read_csv(sample_info['history_file'], header=None)
            if history_df.empty:
                raise ValueError(f"History file is empty: {sample_info['history_file']}")
            history_data = history_df.values.flatten()
            
            # Загружаем target данные с проверкой
            target_df = pd.read_csv(sample_info['target_file'], header=None)
            if target_df.empty:
                raise ValueError(f"Target file is empty: {sample_info['target_file']}")
            target_data = target_df.values.flatten()
            
            # Проверяем, что данные не пустые
            if len(history_data) == 0 or len(target_data) == 0:
                raise ValueError(f"Empty data in files: {sample_info['history_file']} or {sample_info['target_file']}")
            
            # Формируем sample
            sample = {
                'history': torch.FloatTensor(history_data).unsqueeze(0).unsqueeze(2),
                'target': torch.FloatTensor(target_data).unsqueeze(0).unsqueeze(2),
                'ticker': sample_info['ticker']
            }
            
            # Применяем transform если задан
            if self.transform:
                sample = self.transform(sample)
            
            return sample
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            print(f"History file: {sample_info['history_file']}")
            print(f"Target file: {sample_info['target_file']}")
            # Возвращаем None или raise исключение
            raise e