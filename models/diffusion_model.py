import torch
import os
from safetensors.torch import save_file, load_file, load_model, save_model
from diffusers import DDPMScheduler
import json
from typing import Union, Dict, Any
from src import (
    TradingGDTTransformer,
    UNetStockModel,
    CustomDiffusionModel, 
    TradingGDTFeatureExtractor
)


MODEL_REGISTRY = {
    "TradingGDTTransformer": TradingGDTTransformer,
    "UNetStockModel": UNetStockModel,
    "CustomDiffusionModel": CustomDiffusionModel, 
}


class TradingGDTModel:
    def __init__(
        self, 
        device: str = "mps",
    ):
        """
        Инициализация модели TradingGDT.
        
        Args:
            device (str): Устройство для вычислений ('cpu', 'cuda', 'mps').
        """
        self.device = device
        
        # Инициализация scheduler'а
        self.scheduler = DDPMScheduler()
        
        # Инициализация feature extractor'а
        self.extractor = TradingGDTFeatureExtractor(
            input_size=5,
            feature_size=256,
        )
        try:
            load_model(self.extractor, 'pretrained-extractor/trading_feature_extractor.safetensors')
            print("Feature extractor успешно загружен")
        except Exception as e:
            print(f"Предупреждение: Не удалось загрузить feature extractor: {e}")
            
        self.extractor.to(device)
        self.extractor.eval()

        # Конфигурация модели по умолчанию
        self.model_config = None
        # Инициализация модели по умолчанию
        self.backbone = None


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
        if device is None:
            device = cls._determine_device()

        # Загрузка конфигурации
        if isinstance(config, str):
            if not os.path.exists(config):
                raise FileNotFoundError(f"Конфигурационный файл не найден: {config}")
            with open(config, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise TypeError("config должен быть строкой (путь к файлу) или словарем")

        model_type = config_dict.get("model_type")
        if model_type is None:
            raise ValueError("Конфигурация должна содержать ключ 'model_type'")
        
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Неизвестный тип модели: {model_type}. Доступные типы: {list(MODEL_REGISTRY.keys())}")
        
        model_class = MODEL_REGISTRY[model_type]

        # Создание экземпляра класса
        model_instance = cls(device=device)
        model_instance.model_config = config_dict.copy()
        
        # Фильтрация конфигурации для создания модели (исключение служебных полей)
        model_config_filtered = {k: v for k, v in config_dict.items() if k not in ["model_type", "device"]}
        
        # Создание модели
        try:
            model_instance.backbone = model_class(**model_config_filtered).to(device)
        except Exception as e:
            raise RuntimeError(f"Ошибка при создании модели {model_type}: {e}") from e
            
        return model_instance
    

    def save_pretrained(self, dir_path: str = "pretrained-models"):
        """
        Сохраняет модель и связанные компоненты в указанную базовую директорию.
        Создаёт поддиректорию с именем типа модели.
        Структура: {base_dir_path}/{model_type}/config.json и {model_type}.safetensors

        Args:
            base_dir_path (str): Базовый путь к директории для сохранения всех моделей.
                                 По умолчанию "pretrained-models".
        """
        if self.backbone is None:
            raise ValueError("Нет модели для сохранения. Инициализируйте модель сначала.")
        if self.model_config is None or "model_type" not in self.model_config:
            raise ValueError("Конфигурация модели не найдена или не содержит 'model_type'.")

        model_type = self.model_config["model_type"]
        # Создаем поддиректорию для конкретной модели
        model_dir_path = os.path.join(dir_path, model_type)
        os.makedirs(model_dir_path, exist_ok=True)

        # Сохранение конфигурации
        config_path = os.path.join(model_dir_path, "config.json")
        config_to_save = self.model_config.copy()
        config_to_save["device"] = str(self.device)
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Ошибка при сохранении config.json в {config_path}: {e}") from e

        # Сохранение весов модели
        model_weights_path = os.path.join(model_dir_path, f"{model_type}.safetensors")
        try:
            save_file(self.backbone.state_dict(), model_weights_path)
        except Exception as e:
            raise RuntimeError(f"Ошибка при сохранении весов модели в {model_weights_path}: {e}") from e

        print(f"Модель {model_type} успешно сохранена в {model_dir_path}")


    @classmethod
    def from_pretrained(cls, dir_path: str, device: str = None):
        """
        Загружает модель из указанной базовой директории.
        Предполагается структура: {base_dir_path}/{model_type}/config.json и {model_type}.safetensors

        Args:
            base_dir_path (str): Путь к базовой директории, содержащей поддиректории моделей.
            device (str, optional): Устройство для загрузки модели.

        Returns:
            TradingGDTModel: Экземпляр загруженной модели.
        """
        if device is None:
            device = cls._determine_device()

        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Базовая директория моделей не найдена: {dir_path}")

        # Предполагаем, что base_dir_path содержит поддиректорию с именем типа модели
        # Нам нужно найти эту поддиректорию. Мы можем попробовать загрузить config.json
        # из base_dir_path напрямую, если это старый формат, или искать поддиректорию.
        # Простой способ: если base_dir_path содержит config.json, используем его как есть.
        # Иначе предполагаем, что это базовая директория и ищем поддиректорию.

        config_path_direct = os.path.join(dir_path, "config.json")
        model_dir_path = dir_path # По умолчанию, если config.json прямо тут

        if not os.path.exists(config_path_direct):
            # Предполагаем, что base_dir_path - это базовая директория
            # Пытаемся найти поддиректорию с config.json
            try:
                # Пробуем загрузить config, чтобы узнать model_type
                # Это немного неуклюже, но работает
                temp_config_path = None
                # Ищем любую поддиректорию с config.json
                for item in os.listdir(dir_path):
                    item_path = os.path.join(dir_path, item)
                    if os.path.isdir(item_path):
                        possible_config = os.path.join(item_path, "config.json")
                        if os.path.exists(possible_config):
                            temp_config_path = possible_config
                            model_dir_path = item_path
                            break

                if temp_config_path is None:
                     # Пробуем интерпретировать base_dir_path как путь к модели напрямую
                     # (для обратной совместимости)
                     model_dir_path = dir_path
                     # Проверка будет ниже
                     pass
                # Если нашли, model_dir_path уже установлен
            except (OSError, StopIteration):
                # Если не можем прочитать директорию или не находим поддиректорию
                # Пробуем интерпретировать base_dir_path как путь к модели напрямую
                model_dir_path = dir_path
                # Проверка будет ниже

        config_path = os.path.join(model_dir_path, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Файл конфигурации не найден в {model_dir_path} (ожидался {config_path})")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке config.json из {config_path}: {e}") from e

        model_type = config_dict.get("model_type")
        if model_type is None:
            raise ValueError("Конфигурация должна содержать ключ 'model_type'")

        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Неизвестный тип модели: {model_type}. Доступные типы: {list(MODEL_REGISTRY.keys())}")

        model_class = MODEL_REGISTRY[model_type]

        # Создание экземпляра класса
        model_instance = cls(device=device)
        model_instance.model_config = config_dict.copy()

        # Создание модели
        model_config_filtered = {k: v for k, v in config_dict.items() if k not in ["model_type", "device"]}
        try:
            model_instance.backbone = model_class(**model_config_filtered).to(device)
        except Exception as e:
            raise RuntimeError(f"Ошибка при создании модели {model_type}: {e}") from e

        # Загрузка весов модели
        model_weights_path = os.path.join(model_dir_path, f"{model_type}.safetensors")
        # Альтернативный путь для совместимости
        model_weights_path_alt = os.path.join(model_dir_path, "backbone.safetensors")
        weights_path_to_use = model_weights_path
        if not os.path.exists(model_weights_path):
            if os.path.exists(model_weights_path_alt):
                weights_path_to_use = model_weights_path_alt
            else:
                 raise FileNotFoundError(
                    f"Файл весов модели не найден: ожидался {model_weights_path} или {model_weights_path_alt}"
                )

        try:
            state_dict = load_file(weights_path_to_use)
            model_instance.backbone.load_state_dict(state_dict)
            model_instance.backbone.eval()
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке весов модели из {weights_path_to_use}: {e}") from e

        print(f"Модель {model_type} успешно загружена из {model_dir_path}")
        return model_instance


    def to(self, device: Union[str, torch.device]):
        """
        Перемещает модель на указанное устройство.
        
        Args:
            device (str or torch.device): Целевое устройство.
            
        Returns:
            TradingGDTModel: self
        """
        self.device = str(device)
        if self.backbone is not None:
            self.backbone = self.backbone.to(device)
        self.extractor = self.extractor.to(device)
        return self


    def train(self):
        """Переводит модель в режим обучения."""
        if self.backbone is not None:
            self.backbone.train()
        return self


    def eval(self):
        """Переводит модель в режим оценки."""
        if self.backbone is not None:
            self.backbone.eval()
        return self


    def parameters(self):
        """Возвращает параметры модели для оптимизатора."""
        if self.backbone is not None:
            return self.backbone.parameters()
        else:
            raise ValueError("Модель не инициализирована. Вызовите from_config или from_pretrained.")


    @staticmethod
    def _determine_device() -> str:
        """Определяет доступное устройство для вычислений."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        else:
            return "cpu"