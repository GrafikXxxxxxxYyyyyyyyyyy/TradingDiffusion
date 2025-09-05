import torch
from tqdm import tqdm
from typing import Optional, Union

from models.diffusion_model import TradingGDTModel

import numpy as np
import matplotlib.pyplot as plt



def plot_history_and_prediction(history, predict):
    """
    Отрисовывает исторические цены и прогноз на одном графике
    
    Args:
        history: torch.Tensor размерности [1, 256, 1] - исторические данные
        predict: torch.Tensor размерности [1, 32, 1] - прогнозируемые данные
    """
    # Преобразуем тензоры в numpy массивы и убираем лишние размерности
    history_np = history.squeeze().detach().cpu().numpy()
    predict_np = predict.squeeze().detach().cpu().numpy()
    
    # Создаем массивы для оси x
    history_x = np.arange(len(history_np))
    predict_x = np.arange(len(history_np), len(history_np) + len(predict_np))
    
    # Создаем график
    plt.figure(figsize=(12, 6))
    
    # Рисуем исторические данные
    plt.plot(history_x, history_np, label='История', color='blue', linewidth=1)
    
    # Рисуем прогноз
    plt.plot(predict_x, predict_np, label='Прогноз', color='red', linewidth=1)
    
    # Соединяем последнюю точку истории с первой точкой прогноза
    plt.plot([history_x[-1], predict_x[0]], [history_np[-1], predict_np[0]], 
             color='red', linewidth=1, linestyle='--', alpha=0.7)
    
    # Настройки графика
    plt.xlabel('Время')
    plt.ylabel('Цена закрытия')
    plt.title('Исторические цены и прогноз')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Показываем график
    plt.tight_layout()
    plt.show()



# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used,
            `timesteps` must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    timesteps = scheduler.timesteps

    return timesteps, num_inference_steps



class TradingGDTPipeline:
    def __init__(self, device='mps'):
        self.device = device


    def __call__(
        self,
        model: TradingGDTModel,
        ticker: Optional[str] = None,
        history_prices: Optional[torch.FloatTensor] = None,
        num_inference_steps: int = 50,
        batch_size: int = 1,
        generator: Optional[torch.Generator] = None,
    ):  
        self.model = model

        # 1. Получаем исторические цены
        if ticker is not None and history_prices is None:
            # Получаем цены при помощи yfinance
            # ...
            pass
        elif ticker is None and history_prices is not None:
            history_prices = history_prices.to(model.device)
        else:
            raise ("Должны быть переданы либо ticker либо исторические цены")
        print(f"History prices shape: {history_prices.shape}")


        # 2. Процессим исторические цены
        processed_prices = self.model.extractor.extract_features(history_prices)
        processed_prices = processed_prices.to(self.model.device)
        print(f"Processed prices shape: {processed_prices.shape}")


        # 3. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.model.scheduler, num_inference_steps, self.model.device
        )
        print(f"Timesteps: {timesteps}\nNum steps: {num_inference_steps}")


        # 4. Подготавливаем шумный вход для transformer
        shape = (batch_size, 32, 1)
        noisy_input = torch.randn(shape, generator=generator).to(self.model.device)


        # 5. Denoising loop
        for i, t in tqdm(enumerate(timesteps)):
            noisy_input = self.model.scheduler.scale_model_input(noisy_input, t)

            # 5.1 Предсказываем шум моделью
            with torch.no_grad():
                noise_pred = self.model.transformer(
                    noisy_targets=noisy_input,
                    timestep=t,
                    processor_hidden_states=processed_prices,
                )

            # 5.2 рассчитываем предыдущий шумный семпл x_t -> x_t-1
            noisy_input = self.model.scheduler.step(noise_pred, t, noisy_input, return_dict=False)[0]

        return noisy_input