import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union, Optional


class TimestepEmbedding(nn.Module):
    """
    Встраивание временных шагов диффузии с использованием синусоидальных позиционных эмбеддингов.
    """
    def __init__(self, embedding_dim: int, max_period: int = 10000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: тензор временных шагов формы [batch_size] или [batch_size, 1]
            
        Returns:
            эмбеддинги временных шагов формы [batch_size, embedding_dim]
        """
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        exponent = -math.log(self.max_period) * torch.arange(
            half_dim, dtype=torch.float32, device=device
        )
        exponent = exponent / (half_dim - 1)
        emb = torch.exp(exponent)
        
        # Убедимся, что timesteps имеет правильную форму
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(-1)
            
        emb = timesteps.float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Если embedding_dim нечетное, добавляем нулевое значение
        if self.embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
            
        return emb


class MultiheadAttention(nn.Module):
    """
    Упрощённая реализация многоголового внимания.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: тензор формы [batch_size, seq_len_q, embed_dim]
            key: тензор формы [batch_size, seq_len_k, embed_dim]
            value: тензор формы [batch_size, seq_len_k, embed_dim]
            attn_mask: маска внимания (опционально)
            
        Returns:
            выход тензора формы [batch_size, seq_len_q, embed_dim]
        """
        batch_size = query.size(0)
        
        # Проекции
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Вычисление_attention_scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attn_mask is not None:
            attn_scores += attn_mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        
        # Применение_attention
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        return self.out_proj(attn_output)


class DiffusionBlock(nn.Module):
    """
    Базовый блок диффузионной модели с условием.
    """
    def __init__(self, hidden_size: int, condition_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        
        # Нормализация и линейные слои для обработки входов
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(condition_size)
        
        # Внимание между зашумленным таргетом и условием
        self.cross_attention = MultiheadAttention(hidden_size, num_heads, dropout)
        
        # Самовнимание внутри таргета
        self.self_attention = MultiheadAttention(hidden_size, num_heads, dropout)
        
        # Полносвязный блок
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Адаптер для проекции условия в пространство таргета
        self.condition_adapter = nn.Linear(condition_size, hidden_size)
        
    def forward(self, noisy_target: torch.Tensor, timestep_embedding: torch.Tensor, condition: torch.Tensor):
        """
        Args:
            noisy_target: зашумленный таргет формы [batch_size, seq_len_target, hidden_size]
            timestep_embedding: эмбеддинг временного шага формы [batch_size, time_embed_dim]
            condition: условие формы [batch_size, seq_len_condition, condition_size]
            
        Returns:
            обновлённый таргет формы [batch_size, seq_len_target, hidden_size]
        """
        batch_size = noisy_target.size(0)
        
        # Адаптируем условие к размерности таргета
        condition_proj = self.condition_adapter(self.norm2(condition))  # [batch_size, seq_len_condition, hidden_size]
        
        # Кросс-внимание между таргетом и условием
        attn_output = self.cross_attention(
            query=self.norm1(noisy_target),
            key=condition_proj,
            value=condition_proj
        )
        noisy_target = noisy_target + attn_output
        
        # Самовнимание внутри таргета
        self_attn_output = self.self_attention(
            query=self.norm1(noisy_target),
            key=self.norm1(noisy_target),
            value=self.norm1(noisy_target)
        )
        noisy_target = noisy_target + self_attn_output
        
        # MLP
        mlp_output = self.mlp(self.norm1(noisy_target))
        noisy_target = noisy_target + mlp_output
        
        return noisy_target


class CustomDiffusionModel(nn.Module):
    """
    Кастомная диффузионная модель для прогнозирования временных рядов.
    """
    def __init__(
        self,
        target_seq_len: int = 32,
        target_input_dim: int = 1,
        condition_seq_len: int = 256,
        condition_input_dim: int = 256,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        time_embedding_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.target_seq_len = target_seq_len
        self.target_input_dim = target_input_dim
        self.condition_seq_len = condition_seq_len
        self.condition_input_dim = condition_input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Встраивание входов
        self.target_embedding = nn.Linear(target_input_dim, hidden_size)
        self.condition_embedding = nn.Linear(condition_input_dim, condition_input_dim)  # Можно оставить как есть или изменить
        
        # Встраивание временных шагов
        self.time_embedding = TimestepEmbedding(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Стек диффузионных блоков
        self.blocks = nn.ModuleList([
            DiffusionBlock(hidden_size, condition_input_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Выходной слой
        self.norm_out = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, target_input_dim)
        
    def forward(self, noisy_targets: torch.Tensor, timestep: Union[torch.Tensor, int], processor_hidden_states: torch.Tensor):
        """
        Args:
            noisy_target: зашумленный таргет формы [batch_size, target_seq_len, target_input_dim]
            timestep: временной шаг (тензор формы [batch_size] или скаляр)
            condition: условие формы [batch_size, condition_seq_len, condition_input_dim]
            
        Returns:
            предсказание шума формы [batch_size, target_seq_len, target_input_dim]
        """
        # Преобразуем timestep в тензор, если это скаляр
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=noisy_targets.device)
        
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
            
        # Если timestep одномерный, повторяем для каждого элемента в батче
        if timestep.dim() == 1 and timestep.size(0) != noisy_targets.size(0):
            timestep = timestep.repeat(noisy_targets.size(0))
            
        # Встраивание входов
        target_emb = self.target_embedding(noisy_targets)  # [batch_size, target_seq_len, hidden_size]
        condition_emb = self.condition_embedding(processor_hidden_states)  # [batch_size, condition_seq_len, condition_input_dim]
        
        # Встраивание временных шагов
        time_emb = self.time_embedding(timestep)  # [batch_size, time_embedding_dim]
        time_emb = self.time_mlp(time_emb)  # [batch_size, hidden_size]
        
        # Проход через блоки
        hidden = target_emb
        for block in self.blocks:
            hidden = block(hidden, time_emb, condition_emb)
            
        # Выходной слой
        output = self.norm_out(hidden)
        output = self.out_proj(output)
        
        return output