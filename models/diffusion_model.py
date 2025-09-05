# models/gdt/model.py
import torch
import os
from safetensors.torch import save_file, load_file, load_model
from diffusers import DDPMScheduler
import json
from typing import Union, Dict, Any
from src import (
    
)



class TradingGDTModel:
    def __init__(
        self, 
        device: str = "mps",
        model_config: Dict[str, Any] = None
    ):
        """
        Инициализация модели TradingGDT.
        
        Args:
            device (str): Устройство для вычислений ('cpu', 'cuda', 'mps').
            model_config (Dict[str, Any], optional): Конфигурация модели.
        """
        self.device = device
        
        # Конфигурация модели по умолчанию
        self.model_config = model_config or {
            "target_sequence_length": 32,
            "history_sequence_length": 256,
            "target_input_dim": 1,
            "history_input_dim": 256,
            "hidden_size": 512,
            "num_layers": 12,
            "attention_head_dim": 64,
            "num_attention_heads": 8,
            "timestep_embedding_dim": 256
        }

        # Инициализация scheduler'а
        self.scheduler = DDPMScheduler()
        
        # Инициализация feature extractor'а
        self.extractor = TradingFeatureExtractor(
            input_size=5,
            feature_size=256,
        )
        # Попытка загрузить предобученный extractor
        try:
            load_model(self.extractor, 'pretrained-extractor/trading_feature_extractor.safetensors')
            print("Feature extractor успешно загружен")
        except Exception as e:
            print(f"Предупреждение: Не удалось загрузить feature extractor: {e}")
            
        self.extractor.to(device)
        self.extractor.eval()

        # Инициализация трансформера
        self.transformer = TradingGDTTransformer(
            target_sequence_length=self.model_config["target_sequence_length"],
            history_sequence_length=self.model_config["history_sequence_length"],
            target_input_dim=self.model_config["target_input_dim"],
            history_input_dim=self.model_config["history_input_dim"],
            num_layers=self.model_config["num_layers"],
            attention_head_dim=self.model_config["attention_head_dim"],
            num_attention_heads=self.model_config["num_attention_heads"],
            hidden_size=self.model_config["hidden_size"],
            timestep_embedding_dim=self.model_config["timestep_embedding_dim"]
        ).to(device)

        # Подсчет общего количества параметров
        total_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"Общее количество параметров трансформера: {total_params / 1e6:.2f} millions")


    @classmethod
    def from_config(cls, config: Union[str, Dict[str, Any]], device: str = None):
        """
        Создает модель из конфигурационного файла или словаря.
        
        Args:
            config (str or Dict): Путь к конфигурационному файлу или словарь с конфигурацией.
            device (str, optional): Устройство для вычислений.
            
        Returns:
            TradingGDTModel: Экземпляр модели.
        """
        if isinstance(config, str):
            # Если передан путь к файлу
            if not os.path.exists(config):
                raise FileNotFoundError(f"Конфигурационный файл не найден: {config}")
            
            with open(config, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
        elif isinstance(config, dict):
            # Если передан словарь
            model_config = config
        else:
            raise TypeError("config должен быть строкой (путь к файлу) или словарем")
        
        # Определяем устройство
        if device is None:
            device = model_config.get("device", "cpu")
        
        # Создаем экземпляр модели
        return cls(device=device, model_config=model_config)


    @classmethod
    def from_pretrained(cls, dir_path: str, device: str = None):
        """
        Загружает модель из указанной директории.
        
        Args:
            dir_path (str): Путь к директории с сохраненной моделью.
            device (str, optional): Устройство для загрузки модели.
            
        Returns:
            TradingGDTModel: Экземпляр загруженной модели.
        """
        # Загружаем конфигурацию
        config_path = os.path.join(dir_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Определяем устройство
        if device is None:
            device = config.get("device", "cpu")
        
        # Создаем экземпляр модели
        model = cls.__new__(cls)  # Создаем экземпляр без вызова __init__
        model.device = device
        model.model_config = config
        
        # Создаем компоненты
        model.scheduler = DDPMScheduler()
        
        # Создаем трансформер с параметрами из конфига
        model.transformer = TradingGDTTransformer(
            target_sequence_length=config["target_sequence_length"],
            history_sequence_length=config["history_sequence_length"],
            target_input_dim=config["target_input_dim"],
            history_input_dim=config["history_input_dim"],
            num_layers=config["num_layers"],
            attention_head_dim=config["attention_head_dim"],
            num_attention_heads=config["num_attention_heads"],
            hidden_size=config["hidden_size"],
            timestep_embedding_dim=config["timestep_embedding_dim"]
        ).to(device)
        
        # Загружаем веса трансформера
        transformer_path = os.path.join(dir_path, "transformer.safetensors")
        if not os.path.exists(transformer_path):
            raise FileNotFoundError(f"Файл весов не найден: {transformer_path}")
            
        transformer_state_dict = load_file(transformer_path)
        model.transformer.load_state_dict(transformer_state_dict)
        
        # Инициализация feature extractor'а
        model.extractor = TradingFeatureExtractor(
            input_size=5,
            feature_size=256,
        )
        # Попытка загрузить предобученный extractor
        try:
            load_model(model.extractor, 'pretrained-extractor/trading_feature_extractor.safetensors')
            print("Feature extractor успешно загружен")
        except Exception as e:
            print(f"Предупреждение: Не удалось загрузить feature extractor: {e}")
            
        model.extractor.to(device)
        model.extractor.eval()
        
        # Переводим модель в режим оценки
        model.transformer.eval()
        
        print(f"Модель успешно загружена из {dir_path}")
        return model


    def save_pretrained(self, dir_path: str):
        """
        Сохраняет модель и связанные компоненты в указанную директорию в формате safetensors.
        
        Args:
            dir_path (str): Путь к директории для сохранения модели.
        """
        # Создаем директорию если её нет
        os.makedirs(dir_path, exist_ok=True)
        
        # Сохраняем веса трансформера
        transformer_state_dict = self.transformer.state_dict()
        save_file(transformer_state_dict, os.path.join(dir_path, "transformer.safetensors"))
        
        # Сохраняем конфигурацию модели
        config = self.model_config.copy()
        config["device"] = str(self.device)
        
        config_path = os.path.join(dir_path, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"Модель успешно сохранена в {dir_path}")


    def to(self, device: Union[str, torch.device]):
        """
        Перемещает модель на указанное устройство.
        
        Args:
            device (str or torch.device): Целевое устройство.
            
        Returns:
            TradingGDTModel: self
        """
        self.device = str(device)
        self.transformer = self.transformer.to(device)
        self.extractor = self.extractor.to(device)
        return self

    def train(self):
        """Переводит модель в режим обучения."""
        self.transformer.train()
        return self

    def eval(self):
        """Переводит модель в режим оценки."""
        self.transformer.eval()
        return self

    def parameters(self):
        """Возвращает параметры модели для оптимизатора."""
        return self.transformer.parameters()