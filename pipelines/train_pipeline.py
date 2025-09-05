# pipelines/gdt/train_pipeline.py
import os
import math
import torch
from tqdm import tqdm
import torch.nn.functional as F
from typing import Any, Optional, Union, Dict
from dataclasses import dataclass
import time

# --- Добавлено для TensorBoard ---
from torch.utils.tensorboard import SummaryWriter
# --- --- ---

from models.diffusion_model import TradingGDTModel
from src.utils.dataset import TradingDataset


@dataclass
class TradingGDTTrainingArgs:
    seed: Optional[int] = None
    train_batch_size: int = 8
    output_dir: str = 'TradingGDT-pretrained'
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-08
    adam_weight_decay: float = 1e-2
    dataloader_num_workers: int = 0
    num_train_epochs: int = 1
    save_steps: int = 1000
    # --- Добавлено для TensorBoard ---
    tensorboard_log_dir: str = "runs/trading_experiment" # Директория для логов TensorBoard
    # --- --- ---


class TradingGDTTrainer:
    def __init__(
        self, 
        model: TradingGDTModel, 
        args: TradingGDTTrainingArgs,
        train_dataset: TradingDataset,
    ):
        self.model = model
        self.args = args
        self.dataset = train_dataset
        # --- Добавлено для TensorBoard ---
        self.writer: Optional[SummaryWriter] = None
        # --- --- ---

    def train(self):
        # --- Добавлено для TensorBoard: Инициализация SummaryWriter ---
        if self.args.tensorboard_log_dir:
            try:
                # Создаём уникальное имя запуска на основе времени
                timestamp = str(int(time.time()))
                run_name = f"run_{timestamp}"
                full_log_dir = os.path.join(self.args.tensorboard_log_dir, run_name)
                self.writer = SummaryWriter(log_dir=full_log_dir)
                print(f"Инициализирован TensorBoard логгер. Логи будут сохранены в {full_log_dir}")
            except Exception as e:
                print(f"Не удалось инициализировать TensorBoard: {e}. Логирование отключено.")
                self.writer = None
        # --- --- ---

        # 1. Создаём директорию проекта
        if self.args.output_dir is not None:
            os.makedirs(self.args.output_dir, exist_ok=True)

        # 2. Включаем обучение параметров трансформера
        self.model.transformer.train()

        # 3. Initialize the optimizer
        optimizer = torch.optim.AdamW(
            self.model.transformer.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            weight_decay=self.args.adam_weight_decay,
            eps=self.args.adam_epsilon,
        )

        # 4. DataLoaders creation:
        def collate_fn(example):
            histories = [item['history'] for item in example]
            targets = [item['target'] for item in example]
            tickers = [item['ticker'] for item in example]

            batch_histories = torch.cat(histories, dim=0)  # [batch_size, 256, 5]
            batch_targets = torch.cat(targets, dim=0)      # [batch_size, 32, 1]

            return {
                'history': batch_histories,
                'target': batch_targets,
                'ticker': tickers
            }

        train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=self.args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        # Вычисляем общее количество шагов для прогресс-бара
        total_steps = len(train_dataloader) * self.args.num_train_epochs

        # 5. Training loop
        global_step = 0
        progress_bar = tqdm(
            total=total_steps,
            desc="Training",
            leave=True
        )
        
        for epoch in range(self.args.num_train_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for step, batch in enumerate(train_dataloader):
                target = batch['target']
                target = target.to(self.model.device)
                history = batch['history']
                history = history.to(self.model.device)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(target)
                noise = noise.to(self.model.device)

                bsz = target.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, self.model.scheduler.config.num_train_timesteps, (bsz,), device=self.model.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_input = self.model.scheduler.add_noise(target, noise, timesteps)
                noisy_input = noisy_input.to(self.model.device)

                # Get the processed prices for conditioning
                with torch.no_grad():  # Feature extractor в режиме eval, градиенты не нужны
                    processor_hidden_states = self.model.extractor.extract_features(history)
                    processor_hidden_states = processor_hidden_states.to(self.model.device)

                # Predict the noise residual and compute loss
                model_pred = self.model.transformer(
                    noisy_targets=noisy_input,
                    timestep=timesteps,
                    processor_hidden_states=processor_hidden_states,
                )

                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Обновляем метрики
                epoch_loss += loss.detach().item()
                num_batches += 1
                global_step += 1

                # Обновляем прогресс-бар
                progress_bar.set_postfix({
                    'Epoch': f"{epoch+1}/{self.args.num_train_epochs}",
                    'Loss': f"{loss.detach().item():.6f}",
                    'Avg_Loss': f"{epoch_loss/num_batches:.6f}"
                })
                progress_bar.update(1)

                # --- Добавлено для TensorBoard: Логирование метрик ---
                if self.writer is not None:
                    try:
                        self.writer.add_scalar("Loss/step", loss.detach().item(), global_step)
                    except Exception as log_e:
                        print(f"Ошибка при логировании в TensorBoard на шаге {global_step}: {log_e}")
                # --- --- ---

                # Сохраняем модель
                if global_step > 0 and global_step % self.args.save_steps == 0:
                    try:
                        self.model.save_pretrained(dir_path=self.args.output_dir)
                        print(f"Модель сохранена в {self.args.output_dir}")
                    except Exception as e:
                        print(f"Ошибка при сохранении модели: {e}")

            # Логируем среднюю потерю по эпохе
            avg_epoch_loss = epoch_loss / num_batches
            print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.6f}")
            
            # --- Добавлено для TensorBoard: Логирование средней потери по эпохе ---
            if self.writer is not None:
                try:
                    self.writer.add_scalar("Loss/epoch", avg_epoch_loss, epoch + 1)
                except Exception as log_e:
                    print(f"Ошибка при логировании средней потери эпохи в TensorBoard: {log_e}")
            # --- --- ---

        # Закрываем прогресс-бар
        progress_bar.close()
        
        # --- Добавлено для TensorBoard: Закрытие SummaryWriter ---
        if self.writer is not None:
            try:
                self.writer.close()
                print(f"TensorBoard логгер закрыт. Логи сохранены в {self.args.tensorboard_log_dir}")
            except Exception as e:
                print(f"Ошибка при закрытии TensorBoard логгера: {e}")
        # --- --- ---

        # Финальное сохранение модели
        try:
            self.model.save_pretrained(dir_path=self.args.output_dir)
            print(f"Обучение завершено. Финальная модель сохранена в {self.args.output_dir}")
        except Exception as e:
            print(f"Ошибка при финальном сохранении модели: {e}")