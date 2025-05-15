import torch
import wandb
import os
import transformers

assert transformers.__version__ == "4.41.2"

from torch import nn
from torch.nn.functional import linear, embedding
from transformers.models.qwen2.modeling_qwen2 import *
from transformers.modeling_outputs import ModelOutput
from tools.log import main_logger
from dataclasses import dataclass
from tools.global_state import hyper_params, data_cls_reversed_dict, ban_losses, ban_layers
from accelerate import Accelerator


accelerator = Accelerator()


class BigValueFirstLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mseloss = nn.MSELoss(reduction="none")

    def forward(self, output, target):
        return torch.mean(torch.abs(target + 1e-2) * self.mseloss(output, target))


class MSELossV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mseloss = nn.MSELoss(reduction="none")
    
    def forward(self, output, target):
        return self.mseloss(output, target).sum(dim=-1).mean()


class L1LossV2(nn.Module):
    def forward(self, output, target):
        return ((output - target).abs().sum(dim=-1)/50.0).mean()


LOSS_DICT = {
    "mseloss": nn.MSELoss,
    "mseloss_v2": MSELossV2,
    "l1loss": nn.L1Loss,
    "l1loss_v2": L1LossV2,
    # "big_value_first": BigValueFirstLoss
}


class CustomConfig(Qwen2Config):
    def set_custom_kwargs(self, **kwargs):
        # required
        self.target_hidden_size = kwargs["target_hidden_size"]
        self.use_attn_map = kwargs.get("use_attn_map", False)
        self.target_rms_norm_eps = kwargs.get("target_rms_norm_eps", self.rms_norm_eps)
        self.use_aux_loss = kwargs.get("use_aux_loss", True)
        self.use_std_like_attn = kwargs.get("use_std_like_attn", False)
        self.use_logits_loss = kwargs.get("use_logits_loss", True)
        self.use_ntp_loss = kwargs.get("use_ntp_loss", True)
        self.check_data_cls_loss = kwargs.get("check_data_cls_loss", False)
        self.kl_temperature = kwargs.get("kl_temperature", 10.0)
        self.aux_loss_type = kwargs.get("aux_loss_type", "mseloss")
        self.student_attn_from_scratch = kwargs.get("student_attn_from_scratch", False)
        self.tie_word_emb_proj = kwargs.get("tie_word_emb_proj", False)
        self.del_layers = kwargs.get("del_layers", [])
        self.use_in_out_mlp = kwargs.get("use_in_out_mlp", False)
        self.use_all_attn = kwargs.get("use_all_attn", False)


class AllAttn(Qwen2FlashAttention2):
    def __init__(self, config: CustomConfig, layer_idx = None):
        super().__init__(config, layer_idx)
        self.config = config
        self.zoom_q = nn.Linear(config.target_hidden_size, self.hidden_size, bias=False)
        self.zoom_k = nn.Linear(config.target_hidden_size, self.hidden_size, bias=False)
        self.zoom_v = nn.Linear(config.target_hidden_size, self.hidden_size, bias=False)
        self.zoom_down = nn.Linear(self.hidden_size, config.target_hidden_size, bias=False)
        self.mseloss = LOSS_DICT[config.aux_loss_type]()
        self.layer_idx = layer_idx

    def part_forward(self, query_states, key_states, value_states, bsz, q_len, position_ids,
                     past_key_value=None, attention_mask=None):
        output_attentions = False

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
            getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
            and self.config.use_sliding_window
        )
        if use_sliding_windows:
            raise NotImplementedError

        if past_key_value is not None:
            raise NotImplementedError

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def forward(
        self,
        hidden_states,
        compressed_hidden_states,
        loss_dict,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        assert past_key_value is None
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        _up_hs_q = self.q_proj(self.zoom_q(compressed_hidden_states))
        _up_hs_k = self.k_proj(self.zoom_k(compressed_hidden_states))
        _up_hs_v = self.v_proj(self.zoom_v(compressed_hidden_states))

        raw_out, raw_attn_map, _ = self.part_forward(query_states, key_states, value_states, bsz, q_len, position_ids,
                                                     attention_mask=None)
        out, attn_map, _ = self.part_forward(_up_hs_q, _up_hs_k, _up_hs_v, bsz, q_len, position_ids,
                                             attention_mask=None)
        compressed_hidden_states = self.zoom_down(out)

        if "attn-q-sim-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict["attn-q-sim-loss"] = self.mseloss(_up_hs_q, query_states)
        if "attn-k-sim-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict["attn-k-sim-loss"] = self.mseloss(_up_hs_k, key_states)
        if "attn-v-sim-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict["attn-v-sim-loss"] = self.mseloss(_up_hs_v, value_states)
        if "attn-k-sim-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict["attn-out-sim-loss"] = self.mseloss(out, raw_out)

        return raw_out, compressed_hidden_states, raw_attn_map, _, loss_dict
    
    def merge_weight(self):
        self.q_proj.weight.data = (self.q_proj.weight.data @ self.zoom_q.weight.data).contiguous()
        self.k_proj.weight.data = (self.k_proj.weight.data @ self.zoom_k.weight.data).contiguous()
        self.v_proj.weight.data = (self.v_proj.weight.data @ self.zoom_v.weight.data).contiguous()
        self.o_proj.weight.data = (self.zoom_down.weight.data @ self.o_proj.weight.data).contiguous()


class Attn(Qwen2FlashAttention2):
    def __init__(self, config: CustomConfig, layer_idx = None):
        super().__init__(config, layer_idx)
        self.config = config
        self.zoom_up = nn.Linear(config.target_hidden_size, self.hidden_size, bias=False)
        self.zoom_down = nn.Linear(self.hidden_size, config.target_hidden_size, bias=False)
        self.mseloss = LOSS_DICT[config.aux_loss_type]()
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states,
        compressed_hidden_states,
        loss_dict,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
        **kwargs,
    ):
        output_attentions = self.config.use_attn_map
        if output_attentions:
            raise NotImplementedError
        assert attention_mask is None
        assert past_key_value is None
        
        raw_out, raw_attn_map, _ = super().forward(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            # cache_position,
            # **kwargs,
        )
        # assert not torch.isnan(compressed_hidden_states).any(), f"NaN detected in model output in a"
        zoomed_hs = self.zoom_up(compressed_hidden_states)
        out, attn_map, _ = super().forward(
            zoomed_hs,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache,
            # cache_position,
            # **kwargs,
        )
        compressed_hidden_states = self.zoom_down(out)

        # assert not torch.isnan(compressed_hidden_states).any(), f"NaN detected in model output in b"
        if "attn-in-sim-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict["attn-in-sim-loss"] = self.mseloss(zoomed_hs, hidden_states)
        if "attn-out-sim-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict["attn-out-sim-loss"] = self.mseloss(out, raw_out)
        return raw_out, compressed_hidden_states, raw_attn_map, _, loss_dict
    
    def merge_weight(self):
        self.q_proj.weight.data = (self.q_proj.weight.data @ self.zoom_up.weight.data).contiguous()
        self.k_proj.weight.data = (self.k_proj.weight.data @ self.zoom_up.weight.data).contiguous()
        self.v_proj.weight.data = (self.v_proj.weight.data @ self.zoom_up.weight.data).contiguous()
        self.o_proj.weight.data = (self.zoom_down.weight.data @ self.o_proj.weight.data).contiguous()


class MLP(Qwen2MLP):
    def __init__(self, config: CustomConfig, layer_idx=None):
        super().__init__(config)
        self.zoom_up = nn.Linear(self.hidden_size, config.target_hidden_size, bias=False)
        self.zoom_gate = nn.Linear(self.hidden_size, config.target_hidden_size, bias=False)
        self.zoom_down = nn.Linear(self.hidden_size, config.target_hidden_size, bias=False)
        self.mseloss = LOSS_DICT[config.aux_loss_type]()
        self.layer_idx = layer_idx

    def small_forward(self, compressed_x, raw_gate, raw_act_gate, raw_up, raw_x, raw_out, loss_dict: dict):
        Wup = self.zoom_up(self.up_proj.weight)
        Wgate = self.zoom_gate(self.gate_proj.weight)
        Wdown = self.zoom_down(self.down_proj.weight.T).T
        gate = linear(compressed_x, Wgate)
        act_gate = self.act_fn(gate)
        up = linear(compressed_x, Wup)
        down = linear(act_gate * up, Wdown)

        # calculate loss
        if "mlp-gate-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict[f"mlp-gate-loss"] = self.mseloss(gate, raw_gate)
        # loss_dict[f"mlp-act-gate-loss"] = self.mseloss(act_gate, raw_act_gate)
        if "mlp-up-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict[f"mlp-up-loss"] = self.mseloss(up, raw_up)
        # loss_dict[f"mlp-in-loss"] = self.mseloss(compressed_x, self.zoom(raw_x))
        if "mlp-out-loss" not in ban_losses and self.layer_idx not in ban_layers:
            loss_dict[f"mlp-out-loss"] = self.mseloss(down, self.zoom_down(raw_out))

        # print("debug 2", loss_dict)
        return down

    def forward(self, x, compressed_x, loss_dict: dict):
        gate = self.gate_proj(x)
        act_gate = self.act_fn(gate)
        up = self.up_proj(x)
        down = self.down_proj(act_gate * up)

        return down, self.small_forward(compressed_x, gate, act_gate, up, x, down, loss_dict), loss_dict
    
    def merge_weight(self):
        self.gate_proj.weight.data = self.zoom_gate(self.gate_proj.weight.data).contiguous()
        self.up_proj.weight.data = self.zoom_up(self.up_proj.weight.data).contiguous()
        self.down_proj.weight.data = self.zoom_down(self.down_proj.weight.data.T).T.contiguous()


class DebugLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        DebugLlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        # self.first = True

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        mm = torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * mm
        out = self.weight * hidden_states.to(input_dtype)
        # if self.first:
        #     assert torch.abs(torch.mean(self.weight) - 1) < 1e-3, f"{self.weight}"
        #     self.first = False
        return out
    

def reinit_weight(module: nn.Module):
    if type(module) == nn.Linear:
        if module.weight.requires_grad:
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    if type(module) == DebugLlamaRMSNorm:
        if module.weight.requires_grad:
            module.weight.data.fill_(1.0)


class CustomLayer(Qwen2DecoderLayer):
    def __init__(self, config: CustomConfig, layer_idx):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx

        if self.config.use_std_like_attn:
            raise ValueError("Low Performance")
        elif self.config.student_attn_from_scratch:
            raise NotImplementedError
        elif self.config.use_all_attn:
            # print(f"[arch] using all attn")
            self.self_attn = AllAttn(config, layer_idx)
        else:
            # print(f"[arch] using io attn")
            self.self_attn = Attn(config, layer_idx)
        if self.config.use_in_out_mlp:
            raise NotImplementedError
        else:
            # print(f"[arch] using all ffn")
            self.mlp = MLP(config, layer_idx)
        self.target_input_layernorm = DebugLlamaRMSNorm(config.target_hidden_size, eps=config.target_rms_norm_eps)
        self.target_post_attention_layernorm = DebugLlamaRMSNorm(config.target_hidden_size, eps=config.target_rms_norm_eps)

    def forward(
        self,
        hidden_states,
        compressed_hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
    ):
        residual = hidden_states
        compressed_residual = compressed_hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        compressed_hidden_states = self.target_input_layernorm(compressed_hidden_states)

        # loss_dict = {}

        # Self Attention
        hidden_states, compressed_hidden_states, self_attn_weights, present_key_value, loss_dict = self.self_attn(
            hidden_states=hidden_states,
            compressed_hidden_states=compressed_hidden_states,
            loss_dict={},
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states = residual + hidden_states
        compressed_hidden_states = compressed_hidden_states + compressed_residual

        # Fully Connected (MLP)
        residual = hidden_states
        compressed_residual = compressed_hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        compressed_hidden_states = self.target_post_attention_layernorm(compressed_hidden_states)

        hidden_states, compressed_hidden_states, loss_dict = self.mlp(hidden_states, compressed_hidden_states, loss_dict)

        hidden_states = residual + hidden_states
        compressed_hidden_states = compressed_hidden_states + compressed_residual

        # MLP end
        # print("debug 3", loss_dict)
        outputs = (hidden_states, compressed_hidden_states, loss_dict)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
    def merge_weight(self):
        self.input_layernorm.weight.data = self.target_input_layernorm.weight.data
        self.post_attention_layernorm.weight.data = self.target_post_attention_layernorm.weight.data


@dataclass
class IIModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    compressed_hidden_states: torch.FloatTensor = None
    aux_loss: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class Model(Qwen2Model):
    _no_split_modules = ["CustomLayer"]

    def __init__(self, config: CustomConfig):
        super().__init__(config)

        self.zoom = nn.Linear(config.hidden_size, config.target_hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [CustomLayer(config, layer_idx) if layer_idx not in config.del_layers else Qwen2DecoderLayer(config, layer_idx)
             for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_norm = DebugLlamaRMSNorm(config.target_hidden_size, eps=config.target_rms_norm_eps)
        self.cur_step = 0

        self.post_init()

    def merge_weight(self):
        self.embed_tokens.weight.data = self.zoom(self.embed_tokens.weight.data).contiguous()
        self.norm.weight.data = self.target_norm.weight.data

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
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
            raise NotImplementedError

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )
        assert attention_mask is None

        # embed positions
        hidden_states = inputs_embeds
        Wemb = self.zoom(self.embed_tokens.weight).to(device=input_ids.device)
        if os.environ.get("DEBUG", False):
            print("emb token", Wemb[0, :6])
        compressed_hidden_states = embedding(input_ids, Wemb)
        # assert not torch.isnan(compressed_hidden_states).any(), f"NaN detected in model output in af emb"

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        aux_loss = 0

        # set state for logging loss
        grad_acumulation_steps = hyper_params["gradient_accumulation_steps"]
        cur_train_step = None
        if (self.cur_step + 1) % (grad_acumulation_steps * 20) == 0:
            cur_train_step = (self.cur_step + 1) // grad_acumulation_steps
        self.cur_step += 1
        
        for layer_idx, decoder_layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                raise NotImplementedError
            
            if layer_idx not in self.config.del_layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    compressed_hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

                compressed_hidden_states = layer_outputs[1]
                loss_dict = layer_outputs[2]
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            
            hidden_states = layer_outputs[0]

            if layer_idx not in self.config.del_layers:
                _log_dict = {}
                for k, v in loss_dict.items():
                    if self.config.use_aux_loss:
                        if isinstance(aux_loss, torch.Tensor):
                            aux_loss = aux_loss.to(v.device)
                        aux_loss = aux_loss + v * hyper_params["aux_loss_scale_factor"]
                    main_logger.debug(f"L{decoder_layer.layer_idx}-{k}: {v.item()}")
                    
                    if cur_train_step:
                        _log_dict[f"L{decoder_layer.layer_idx}-{k}"] = v.item()
                
                if cur_train_step and (os.environ.get("LOCAL_RANK", 0) == 0 or accelerator.is_main_process) and len(_log_dict) > 0:
                    wandb.log(_log_dict, cur_train_step)

        hidden_states = self.norm(hidden_states)
        compressed_hidden_states = self.target_norm(compressed_hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, all_self_attns] if v is not None)
        
        return IIModelOutput(
            last_hidden_state=hidden_states,
            compressed_hidden_states=compressed_hidden_states,
            aux_loss=aux_loss,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


def calculate_language_loss(lgts, labels, vocab_size):
    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = lgts[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
    return loss


class CoTrainLM(Qwen2ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]
    def __init__(self, config: CustomConfig):
        super().__init__(config)
        self.model = Model(config)
        # self.zoom_up = nn.Linear(config.target_hidden_size, config.hidden_size, bias=False)
        if not config.tie_word_emb_proj:
            self.zoom_down = nn.Linear(config.hidden_size, config.target_hidden_size, bias=False)
            self.zoom_down.weight.data.normal_(mean=0.0, std=0.01)  # no init weights
        self.mseloss = LOSS_DICT[config.aux_loss_type]()
        self.kl_temperature = self.config.kl_temperature
        self.cur_step = 0
        self.cur_loss_accumulation = 0
        self.cur_logit_loss_accumulation = 0
        self.check_data_cls_loss = config.check_data_cls_loss
        self.data_cls_losses = [0] * 8
        self.data_cls_cnt = [0] * 8
        self.post_init()

    def merge_weight(self):
        # print(self.lm_head.weight.data.shape)
        if not self.config.tie_word_emb_proj:
            self.lm_head.weight.data = self.zoom_down(self.lm_head.weight.data).contiguous()
        else:
            self.lm_head.weight.data = self.model.zoom(self.lm_head.weight.data).contiguous()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        data_cls=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        compressed_hidden_states = outputs[1]
        aux_loss = outputs[2]

        logits = self.lm_head(hidden_states)
        if not self.config.tie_word_emb_proj:
            Whead = self.zoom_down(self.lm_head.weight)
        else:
            Whead = self.model.zoom(self.lm_head.weight)
        if os.environ.get("DEBUG", False):
            print("head weight", Whead[0, :6])
        target_logits = linear(compressed_hidden_states, Whead)

        if self.config.use_logits_loss:
            target_logp = F.log_softmax(target_logits / self.kl_temperature, dim=-1)
            raw_logp = F.log_softmax(logits / self.kl_temperature, dim=-1)
            logits_loss = F.kl_div(target_logp, raw_logp, log_target=True, reduction="batchmean")
            # logits_loss = self.mseloss(target_logits, logits)
            aux_loss = aux_loss + logits_loss
            main_logger.debug(f"logits_loss: {round(logits_loss.item(), 4)}")
        
        raw_loss = calculate_language_loss(logits.float(), labels, self.config.vocab_size)
        target_loss = calculate_language_loss(target_logits.float(), labels, self.config.vocab_size)
        main_logger.debug(f"raw_loss: {round(raw_loss.item(), 4)}, target_loss: {round(target_loss.item(), 4)}")

        # wandb log
        self.cur_loss_accumulation += target_loss.item()
        if self.config.use_logits_loss:
            self.cur_logit_loss_accumulation += logits_loss.item()
        loss_log_steps = hyper_params["gradient_accumulation_steps"] * 5
        if self.check_data_cls_loss:
            assert hidden_states.shape[0] == 1, "only appliable in bs = 1"
            spec_cls = data_cls[0].item()
            self.data_cls_cnt[spec_cls] += 1
            self.data_cls_losses[spec_cls] += target_loss.item()
        if (self.cur_step + 1) % loss_log_steps == 0:
            cur_train_step = (self.cur_step + 1) // hyper_params["gradient_accumulation_steps"]
            _log_dict = {"target_loss": self.cur_loss_accumulation / loss_log_steps}
            if self.config.use_logits_loss:
                _log_dict["logits_loss"] = self.cur_logit_loss_accumulation / loss_log_steps
            # self.kl_temperature = 0.9 * self.kl_temperature + 0.1 * _log_dict["target_loss"] * 1.5
            if self.check_data_cls_loss:
                _log_dict.update({
                    f"{data_cls_reversed_dict[i]}_loss": loss / self.data_cls_cnt[i] 
                    for i, loss in enumerate(self.data_cls_losses) if self.data_cls_cnt[i] > 0
                })
            if (os.environ.get("LOCAL_RANK", 0) == 0 or accelerator.is_main_process):
                wandb.log(_log_dict, step=cur_train_step)
            self.cur_loss_accumulation = 0
            self.cur_logit_loss_accumulation = 0
            self.data_cls_cnt = [0] * 8
            self.data_cls_losses = [0] * 8
        self.cur_step += 1

        if not return_dict:
            raise NotImplementedError

        return CausalLMOutputWithPast(
            loss=target_loss + aux_loss if self.config.use_ntp_loss else aux_loss,
            logits=target_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def freeze_original_model(self):
        key_words = ["zoom", "target"]
        for n, p in self.named_parameters():
            flag = False

            for key in key_words:
                if key in n:
                    flag = True

            p.requires_grad = flag

    def tie_custom_weights(self, tie_n):
        raise ValueError("low perf")
        layers = self.model.layers
        for i in range(2, self.config.num_hidden_layers - 1, tie_n):
            share_layer: CustomLayer = layers[i]
            for j in range(i + 1, min(self.config.num_hidden_layers - 1, i + tie_n)):
                cur_layer: CustomLayer = layers[j]
                cur_layer.mlp.zoom.weight = share_layer.mlp.zoom.weight
                cur_layer.self_attn.zoom_up.weight = share_layer.self_attn.zoom_up.weight
                cur_layer.self_attn.zoom_down.weight = share_layer.self_attn.zoom_down.weight

    def tie_word_emb_proj(self):
        # self.model.zoom.weight = self.zoom_down.weight
        self.zoom_down.weight = self.model.zoom.weight

    def get_trained_params(self):
        state_dict = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                state_dict[n] = p
        return state_dict

    def save_pretrained(self, *args, **kwargs):
        if kwargs.get("only_save_trainable", True):
            state_dict = self.get_trained_params()
            kwargs["state_dict"] = state_dict
        return super().save_pretrained(*args, **kwargs)
