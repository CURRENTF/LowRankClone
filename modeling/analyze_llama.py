import torch
import wandb
import os

from torch import nn
from torch.nn.functional import linear, embedding
from transformers.models.llama.modeling_llama import *
from transformers.modeling_outputs import ModelOutput
from tools.log import main_logger
from dataclasses import dataclass
from tools.global_state import hyper_params, data_cls_reversed_dict
from accelerate import Accelerator


accelerator = Accelerator()
activations = {}
corr_samples = {}
corr_eff_avg = {}
hyper_params["forward_times"] = 0
hyper_params["sim_mat"] = None


class CustomConfig(LlamaConfig):
    def set_custom_kwargs(self, **kwargs):
        # required
        pass


class Attn(LlamaFlashAttention2):
    def __init__(self, config: CustomConfig, layer_idx = None):
        super().__init__(config, layer_idx)
        self.config = config


class MLP(LlamaMLP):
    def __init__(self, config: CustomConfig, layer_idx):
        super().__init__(config)
        self.mid_avg = torch.zeros(config.intermediate_size, dtype=torch.float32, device="cuda:0")
        self.mid_buffer = torch.zeros(0, config.intermediate_size, dtype=torch.bfloat16, device="cuda:0")
        self.cnt = 0
        self.layer_idx = layer_idx

    def forward(self, x):
        gate = self.gate_proj(x)
        act_gate = self.act_fn(gate)
        up = self.up_proj(x)
        mid_val = act_gate * up
        down = self.down_proj(mid_val)
        
        data_cls = os.environ["DataCls"]

        self.mid_avg = self.mid_avg * self.cnt / (self.cnt + 1) + mid_val.mean(dim=1).mean(dim=0) / (self.cnt + 1)
        self.mid_buffer = torch.cat([self.mid_buffer, mid_val[0, -20:]], dim=0)
        
        activations[f"{data_cls}-L{self.layer_idx:03}"] = self.mid_avg
        corr_samples[f"{data_cls}-L{self.layer_idx:03}"] = self.mid_buffer
        mid_val = mid_val[0].T
        if f"{data_cls}-L{self.layer_idx:03}" not in corr_eff_avg:
            corr_eff_avg[f"{data_cls}-L{self.layer_idx:03}"] = torch.corrcoef(mid_val)
        else:
            corr_eff_avg[f"{data_cls}-L{self.layer_idx:03}"] =\
                corr_eff_avg[f"{data_cls}-L{self.layer_idx:03}"] * self.cnt / (self.cnt + 1) + torch.corrcoef(mid_val) / (self.cnt + 1)
        
        self.cnt += 1

        return down


class CustomLayer(LlamaDecoderLayer):
    def __init__(self, config: CustomConfig, layer_idx):
        super().__init__(config, layer_idx)
        self.config = config
        # self.self_attn = Attn(config, layer_idx)
        # self.mlp = MLP(config, layer_idx)


class Model(LlamaModel):
    _no_split_modules = ["CustomLayer"]

    def __init__(self, config: CustomConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [CustomLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        bf_tensors = []
        sim_mat = torch.zeros(len(self.layers), len(self.layers), device="cuda:0")
        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]
            for j, bf_t in enumerate(bf_tensors):
                sim = torch.cosine_similarity(bf_t, hidden_states, dim=-1).mean()
                sim_mat[j, i] = sim.item()
            bf_tensors.append(hidden_states)

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hyper_params["forward_times"] += 1
        if hyper_params["sim_mat"] is None:
            hyper_params["sim_mat"] = sim_mat[None]
        else:
            hyper_params["sim_mat"] = torch.cat([hyper_params["sim_mat"], sim_mat[None]], dim=0)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class AnalyzeLM(LlamaForCausalLM):
    def __init__(self, config: CustomConfig):
        super().__init__(config)
        self.model = Model(config)
        self.post_init()
