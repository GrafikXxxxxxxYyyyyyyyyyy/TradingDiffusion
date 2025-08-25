import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """
    Создание синусоидальных эмбеддингов временных шагов.
    
    Args:
        timesteps (torch.Tensor): 1-D тензор с временными шагами [N]
        embedding_dim (int): размерность выходных эмбеддингов
        flip_sin_to_cos (bool): порядок конкатенации cos, sin или sin, cos
        downscale_freq_shift (float): сдвиг частот
        scale (float): масштабирующий коэффициент
        max_period (int): максимальная частота
        
    Returns:
        torch.Tensor: тензор эмбеддингов формы [N x dim]
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(timesteps.dtype)
    emb = timesteps[:, None].float() * emb[None, :]

    # масштабирование эмбеддингов
    emb = scale * emb

    # конкатенация синуса и косинуса
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # изменение порядка синуса и косинуса
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # дополнение нулями при нечетной размерности
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb



class TradingGDTDoubleStreamAttnProcessor2_0:
    """
    Процессор внимания для двойного потока архитектуры TradingGDT.
    Реализует совместное вычисление внимания, где потоки исторических данных и целевых данных обрабатываются вместе.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "TradingGDTDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Целевой поток (noisy targets) [B, S_target, D]
        encoder_hidden_states: torch.FloatTensor = None,  # Исторический поток (processed history) [B, S_history, D]
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Вызов процессора совместного внимания.
        
        Args:
            attn: слой внимания
            hidden_states: целевой поток [B, S_target=32, D]
            encoder_hidden_states: исторический поток [B, S_history=256, D]
            attention_mask: маска внимания (опционально)
            
        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: (целевой выход [B, S_target, D], исторический выход [B, S_history, D])
        """
        if encoder_hidden_states is None:
            raise ValueError("TradingGDTDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (history stream)")

        seq_history = encoder_hidden_states.shape[1]  # 256

        # Вычисление QKV для целевого потока (проекции образца)
        target_query = attn.to_q(hidden_states)    # [B, S_target, D]
        target_key = attn.to_k(hidden_states)      # [B, S_target, D]
        target_value = attn.to_v(hidden_states)    # [B, S_target, D]

        # Вычисление QKV для исторического потока (проекции контекста)
        history_query = attn.add_q_proj(encoder_hidden_states)  # [B, S_history, D]
        history_key = attn.add_k_proj(encoder_hidden_states)    # [B, S_history, D]
        history_value = attn.add_v_proj(encoder_hidden_states)  # [B, S_history, D]

        # Преобразование для многоголового внимания
        # [B, S, D] -> [B, S, H, D/H]
        target_query = target_query.unflatten(-1, (attn.heads, -1))
        target_key = target_key.unflatten(-1, (attn.heads, -1))
        target_value = target_value.unflatten(-1, (attn.heads, -1))

        history_query = history_query.unflatten(-1, (attn.heads, -1))
        history_key = history_key.unflatten(-1, (attn.heads, -1))
        history_value = history_value.unflatten(-1, (attn.heads, -1))

        # Применение нормализации QK (если включена)
        if attn.norm_q is not None:
            target_query = attn.norm_q(target_query)
        if attn.norm_k is not None:
            target_key = attn.norm_k(target_key)
        if attn.norm_added_q is not None:
            history_query = attn.norm_added_q(history_query)
        if attn.norm_added_k is not None:
            history_key = attn.norm_added_k(history_key)

        # Объединение для совместного внимания
        # Порядок: [история, цель]
        joint_query = torch.cat([history_query, target_query], dim=1)  # [B, S_history + S_target, H, D/H]
        joint_key = torch.cat([history_key, target_key], dim=1)       # [B, S_history + S_target, H, D/H]
        joint_value = torch.cat([history_value, target_value], dim=1) # [B, S_history + S_target, H, D/H]

        # Вычисление совместного внимания
        # Используем встроенную функцию PyTorch для scaled dot-product attention
        joint_hidden_states = F.scaled_dot_product_attention(
            joint_query.transpose(1, 2),  # [B, H, S_total, D/H]
            joint_key.transpose(1, 2),    # [B, H, S_total, D/H]
            joint_value.transpose(1, 2),  # [B, H, S_total, D/H]
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        ).transpose(1, 2)  # [B, S_total, H, D/H]

        # Преобразование обратно
        joint_hidden_states = joint_hidden_states.flatten(2, 3)  # [B, S_total, D]
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Разделение выходов внимания обратно на два потока
        history_attn_output = joint_hidden_states[:, :seq_history, :]  # Историческая часть [B, S_history, D]
        target_attn_output = joint_hidden_states[:, seq_history:, :]   # Целевая часть [B, S_target, D]

        # Применение выходных проекций
        target_attn_output = attn.to_out[0](target_attn_output)  # [B, S_target, D]
        if len(attn.to_out) > 1:
            target_attn_output = attn.to_out[1](target_attn_output)  # dropout

        history_attn_output = attn.to_add_out(history_attn_output)  # [B, S_history, D]

        return target_attn_output, history_attn_output



class TradingGDTTransformerBlock(nn.Module):
    """
    Блок трансформера TradingGDT с двойным потоком и совместным вниманием.
    """
    
    def __init__(
        self, 
        dim: int,  # размерность скрытого состояния
        num_attention_heads: int,  # количество голов внимания
        attention_head_dim: int,  # размерность каждой головы внимания
        qk_norm: str = "rms_norm",  # тип нормализации QK
        eps: float = 1e-6  # эпсилон для нормализации
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Модули обработки целевого потока (noisy targets)
        # Генерация параметров модуляции (shift, scale, gate) для norm1 и norm2
        self.target_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # 6 * dim для двух наборов параметров (norm1 и norm2)
        )
        self.target_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        # Слой внимания с процессором двойного потока
        self.attn = Attention(
            query_dim=dim,  # размерность запросов целевого потока
            cross_attention_dim=None,  # Включает cross attention для совместного вычисления
            added_kv_proj_dim=dim,  # Включает дополнительные проекции KV для исторического потока
            dim_head=attention_head_dim,  # размерность головы
            heads=num_attention_heads,  # количество голов
            out_dim=dim,  # размерность выхода
            context_pre_only=False,  # позволяет возвращать выходы для обоих потоков
            bias=True,  # использовать смещение
            processor=TradingGDTDoubleStreamAttnProcessor2_0(),  # наш кастомный процессор
            qk_norm=qk_norm,  # тип нормализации QK
            eps=eps,  # эпсилон для нормализации
        )
        self.target_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.target_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Модули обработки исторического потока (processed history)
        # Генерация параметров модуляции для исторического потока
        self.history_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # 6 * dim для двух наборов параметров
        )
        self.history_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        # Исторический поток не нуждается в отдельном слое внимания - он обрабатывается совместно
        self.history_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.history_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def _modulate(self, x, mod_params):
        """
        Применение модуляции к входному тензору.
        
        Args:
            x: входной тензор [B, S, D]
            mod_params: параметры модуляции [B, 3*D] (shift, scale, gate)
            
        Returns:
            Tuple[модулированный тензор, gate]: ([B, S, D], [B, 1, D])
        """
        shift, scale, gate = mod_params.chunk(3, dim=-1)  # Разделяем на 3 части
        # Применяем модуляцию: x * (1 + scale) + shift
        modulated = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        # Gate используется для взвешенного сложения с residual connection
        return modulated, gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,  # Целевой поток (noisy targets) [B, 32, D]
        encoder_hidden_states: torch.Tensor,  # Исторический поток (processed history) [B, 256, D]
        temb: torch.Tensor,  # Временной эмбеддинг [B, D]
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход блока трансформера.
        
        Args:
            hidden_states: целевой поток [B, 32, D]
            encoder_hidden_states: исторический поток [B, 256, D]
            temb: временной эмбеддинг [B, D]
            attention_kwargs: дополнительные аргументы для внимания
            
        Returns:
            Tuple[обновленный исторический поток, обновленный целевой поток]
            - encoder_hidden_states: [B, 256, D]
            - hidden_states: [B, 32, D]
        """
        # Получение параметров модуляции для обоих потоков
        target_mod_params = self.target_mod(temb)    # [B, 6*D]
        history_mod_params = self.history_mod(temb)  # [B, 6*D]

        # Разделение параметров модуляции для norm1 и norm2
        # Каждый поток получает 2 набора параметров: для norm1 и norm2
        target_mod1, target_mod2 = target_mod_params.chunk(2, dim=-1)  # Каждый [B, 3*D]
        history_mod1, history_mod2 = history_mod_params.chunk(2, dim=-1)  # Каждый [B, 3*D]

        # Обработка целевого потока - нормализация 1 + модуляция
        target_normed = self.target_norm1(hidden_states)  # [B, 32, D]
        target_modulated, target_gate1 = self._modulate(target_normed, target_mod1)  # [B, 32, D], [B, 1, D]

        # Обработка исторического потока - нормализация 1 + модуляция
        history_normed = self.history_norm1(encoder_hidden_states)  # [B, 256, D]
        history_modulated, history_gate1 = self._modulate(history_normed, history_mod1)  # [B, 256, D], [B, 1, D]

        # Использование TradingGDTDoubleStreamAttnProcessor2_0 для совместного вычисления внимания
        # Это реализует логику двойного потока:
        # 1. Вычисляет QKV для обоих потоков
        # 2. Применяет нормализацию QK
        # 3. Объединяет и выполняет совместное внимание
        # 4. Разделяет результаты обратно на два потока
        attention_kwargs = attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=target_modulated,        # Целевой поток (как "образец")
            encoder_hidden_states=history_modulated, # Исторический поток (как "контекст")
            **attention_kwargs,
        )

        # TradingGDTDoubleStreamAttnProcessor2_0 возвращает (target_output, history_output)
        target_attn_output, history_attn_output = attn_output

        # Применение ворот внимания и добавление residual (как в Megatron)
        # Обновляем целевой поток: hidden_states + gate1 * attention_output
        hidden_states = hidden_states + target_gate1 * target_attn_output      # [B, 32, D]
        # Обновляем исторический поток: encoder_hidden_states + gate1 * attention_output
        encoder_hidden_states = encoder_hidden_states + history_gate1 * history_attn_output  # [B, 256, D]

        # Обработка целевого потока - нормализация 2 + MLP
        target_normed2 = self.target_norm2(hidden_states)  # [B, 32, D]
        target_modulated2, target_gate2 = self._modulate(target_normed2, target_mod2)  # [B, 32, D], [B, 1, D]
        target_mlp_output = self.target_mlp(target_modulated2)  # [B, 32, D]
        hidden_states = hidden_states + target_gate2 * target_mlp_output  # [B, 32, D]

        # Обработка исторического потока - нормализация 2 + MLP
        history_normed2 = self.history_norm2(encoder_hidden_states)  # [B, 256, D]
        history_modulated2, history_gate2 = self._modulate(history_normed2, history_mod2)  # [B, 256, D], [B, 1, D]
        history_mlp_output = self.history_mlp(history_modulated2)  # [B, 256, D]
        encoder_hidden_states = encoder_hidden_states + history_gate2 * history_mlp_output  # [B, 256, D]

        # Ограничение значений для предотвращения переполнения в fp16
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states



class TradingGDTTransformer(ModelMixin):
    """
    Трансформер TradingGDT для предсказания цен на фондовом рынке.
    
    Эта модель адаптирована из QwenImageTransformer2DModel с сохранением ключевых элементов:
    - Совместное внимание (joint attention) между целевыми и историческими данными
    - Адаптивная модуляция с AdaLayerNormContinuous
    - Двойной поток обработки информации
    
    Args:
        target_sequence_length (`int`, defaults to `32`):
            Длина последовательности целевых данных (таргетов).
        history_sequence_length (`int`, defaults to `256`):
            Длина последовательности исторических данных.
        target_input_dim (`int`, defaults to `1`):
            Размерность входных целевых данных (обычно цена закрытия).
        history_input_dim (`int`, defaults to `17`):
            Размерность входных исторических данных (технические индикаторы).
        hidden_size (`int`, defaults to `512`):
            Размерность скрытого состояния.
        num_layers (`int`, defaults to `20`):
            Количество блоков трансформера.
        attention_head_dim (`int`, defaults to `64`):
            Размерность каждой головы внимания.
        num_attention_heads (`int`, defaults to `8`):
            Количество голов внимания.
        timestep_embedding_dim (`int`, defaults to `256`):
            Размерность эмбеддинга временного шага.
    """

    def __init__(
        self,
        target_sequence_length: int = 32,
        history_sequence_length: int = 256,
        target_input_dim: int = 1,
        history_input_dim: int = 32,
        hidden_size: int = 512,
        num_layers: int = 12,
        attention_head_dim: int = 64,
        num_attention_heads: int = 8,
        timestep_embedding_dim: int = 256,
    ):
        super().__init__()
        
        self.target_sequence_length = target_sequence_length
        self.history_sequence_length = history_sequence_length
        self.target_input_dim = target_input_dim
        self.history_input_dim = history_input_dim
        self.hidden_size = hidden_size
        self.inner_dim = num_attention_heads * attention_head_dim

        # Временной эмбеддинг
        self.time_proj = Timesteps(
            num_channels=timestep_embedding_dim, 
            flip_sin_to_cos=True, 
            downscale_freq_shift=0, 
            scale=1000
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=timestep_embedding_dim, 
            time_embed_dim=hidden_size
        )

        # Проекции входных данных в скрытое пространство
        self.target_in = nn.Linear(target_input_dim, hidden_size)      # [B, 32, 1] -> [B, 32, hidden_size]
        self.history_in = nn.Linear(history_input_dim, hidden_size)    # [B, 256, 17] -> [B, 256, hidden_size]

        # Блоки трансформера
        self.transformer_blocks = nn.ModuleList(
            [
                TradingGDTTransformerBlock(
                    dim=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Нормализация выхода с адаптивной модуляцией
        self.norm_out = AdaLayerNormContinuous(hidden_size, hidden_size, elementwise_affine=False, eps=1e-6)
        # Проекция в пространство целевых данных
        self.proj_out = nn.Linear(hidden_size, target_input_dim, bias=True)  # [B, 32, hidden_size] -> [B, 32, 1]

        self.gradient_checkpointing = False

    def forward(
        self,
        noisy_targets: torch.Tensor,  # Шумные целевые данные [batch_size, 32, 1]
        timestep: Union[torch.LongTensor, int],  # Временной шаг [batch_size] или скаляр
        processor_hidden_states: torch.Tensor,  # Обработанные исторические данные [batch_size, 256, 17]
    ) -> torch.Tensor:
        """
        Прямой проход трансформера TradingGDT.
        
        Args:
            noisy_targets (`torch.Tensor` of shape `(batch_size, 32, 1)`):
                Шумные целевые последовательности цен.
            timestep (`torch.LongTensor` or int):
                Временной шаг диффузии.
            processor_hidden_states (`torch.Tensor` of shape `(batch_size, 256, 17)`):
                Обработанные технические индикаторы.
                
        Returns:
            torch.Tensor: Предсказанный шум для удаления из целевых данных [batch_size, 32, 1]
        """
        # Преобразование timestep в тензор если это скаляр
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=noisy_targets.device)
        
        # Если timestep одномерный, повторяем его для каждого элемента в батче
        if timestep.ndim == 0:
            timestep = timestep.repeat(noisy_targets.shape[0])
        
        # Проекция входных данных в скрытое пространство
        # Целевые данные: [B, 32, 1] -> [B, 32, hidden_size]
        target_hidden_states = self.target_in(noisy_targets)
        # Исторические данные: [B, 256, 17] -> [B, 256, hidden_size]
        history_hidden_states = self.history_in(processor_hidden_states)

        # Создание временных эмбеддингов
        timesteps_proj = self.time_proj(timestep)  # [B, timestep_embedding_dim]
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=target_hidden_states.dtype))  # [B, hidden_size]
        temb = timesteps_emb

        # Проход через блоки трансформера с двойным потоком
        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # Использование gradient checkpointing для экономии памяти
                history_hidden_states, target_hidden_states = self._gradient_checkpointing_func(
                    block,
                    target_hidden_states,
                    history_hidden_states,
                    temb,
                )
            else:
                # Нормальный проход через блок
                history_hidden_states, target_hidden_states = block(
                    hidden_states=target_hidden_states,        # Целевой поток [B, 32, hidden_size]
                    encoder_hidden_states=history_hidden_states, # Исторический поток [B, 256, hidden_size]
                    temb=temb,  # Временной эмбеддинг [B, hidden_size]
                )

        # Используем только целевой поток (target_hidden_states) для финального выхода
        # Применяем адаптивную нормализацию с модуляцией по временному эмбеддингу
        target_hidden_states = self.norm_out(target_hidden_states, temb)  # [B, 32, hidden_size]
        # Проекция в пространство целевых данных
        output = self.proj_out(target_hidden_states)  # [B, 32, 1]

        return output