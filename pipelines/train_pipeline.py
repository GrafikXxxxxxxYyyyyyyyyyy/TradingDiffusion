import os
import math
import torch
from tqdm import tqdm
import torch.nn.functional as F
from typing import Any, Optional, Union, Dict
from dataclasses import dataclass

from models.diffusion_model import TradingGDTModel
from src.utils.dataset import TradingGDTDataset



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



class TradingGDTTrainer:
    def __init__(
        self, 
        model: TradingGDTModel, 
        args: TradingGDTTrainingArgs,
        train_dataset: TradingGDTDataset,
    ):
        self.model = model
        self.args = args
        self.dataset = train_dataset


    def train(self):
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

            batch_histories = torch.cat(histories, dim=0)  # [batch_size, 256, 1]
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


        # 5. Training loop
        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                target = batch['target']
                target = target.to(self.model.device)
                history = batch['history']

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
                processor_hidden_states = self.model.processor(history)
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

                print(f"Epoch: {epoch+1} | Step: {step+1} | Loss: {loss.detach().item()}")

            self.model.save_pretrained(dir_path=self.args.output_dir)