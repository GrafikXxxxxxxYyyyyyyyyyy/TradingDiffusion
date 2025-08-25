import torch
import os
from safetensors.torch import save_file, load_file
from diffusers import DDPMScheduler
from src.processor.price_processor_big import TradingGDTProcessor
from src.transformer.mmdit import TradingGDTTransformer


class TradingGDTModel:
    def __init__(
        self, 
        device: str = "mps"
    ):
        self.device = device

        # Сохраняем параметры для последующего использования
        self.model_config = {
            "target_sequence_length": 32,
            "history_sequence_length": 256,
            "target_input_dim": 1,
            "history_input_dim": 32,
            "hidden_size": 512,
            "num_layers": 12,
            "attention_head_dim": 64,
            "num_attention_heads": 8,
            "timestep_embedding_dim": 256
        }

        self.scheduler = DDPMScheduler()
        self.processor = TradingGDTProcessor()
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
        print(f"Общее количество параметров: {total_params / 1e6:.2f} millions")


    def save_pretrained(self, dir_path):
        """
        Сохраняет модель и связанные компоненты в указанную директорию в формате safetensors
        
        Args:
            dir_path (str): Путь к директории для сохранения модели
        """
        # Создаем директорию если её нет
        os.makedirs(dir_path, exist_ok=True)
        
        # Сохраняем веса трансформера
        transformer_state_dict = self.transformer.state_dict()
        save_file(transformer_state_dict, os.path.join(dir_path, "transformer.safetensors"))
        
        # Сохраняем конфигурацию модели
        config = {
            "target_sequence_length": self.model_config["target_sequence_length"],
            "history_sequence_length": self.model_config["history_sequence_length"],
            "target_input_dim": self.model_config["target_input_dim"],
            "history_input_dim": self.model_config["history_input_dim"],
            "hidden_size": self.model_config["hidden_size"],
            "num_layers": self.model_config["num_layers"],
            "attention_head_dim": self.model_config["attention_head_dim"],
            "num_attention_heads": self.model_config["num_attention_heads"],
            "timestep_embedding_dim": self.model_config["timestep_embedding_dim"],
            "device": str(self.device)
        }
        
        config_path = os.path.join(dir_path, "config.json")
        import json
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"Модель успешно сохранена в {dir_path}")


    @classmethod
    def from_pretrained(cls, dir_path, device=None):
        """
        Загружает модель из указанной директории
        
        Args:
            dir_path (str): Путь к директории с сохраненной моделью
            device (str, optional): Устройство для загрузки модели
            
        Returns:
            TradingGDTModel: Экземпляр загруженной модели
        """
        import json
        
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
        model.processor = TradingGDTProcessor()
        
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
        
        # Переводим модель в режим оценки
        model.transformer.eval()
        
        print(f"Модель успешно загружена из {dir_path}")
        return model


    def to(self, device):
        """
        Перемещает модель на указанное устройство
        
        Args:
            device (str or torch.device): Целевое устройство
            
        Returns:
            TradingGDTModel: self
        """
        self.device = device
        self.transformer = self.transformer.to(device)
        return self


    def train(self):
        """Переводит модель в режим обучения"""
        self.transformer.train()
        return self


    def eval(self):
        """Переводит модель в режим оценки"""
        self.transformer.eval()
        return self