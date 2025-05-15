from transformers.models.llama.modeling_llama import *


class DebugConfig(LlamaConfig):
    def set_custom_kwargs(self, **kwargs):
        # required
        self.skip_start_idx = kwargs["skip_start_idx"]


class DebugLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx


class DebugModel(LlamaModel):
    def __init__(self, config: DebugConfig):
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [DebugLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

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

        for i, decoder_layer in enumerate(self.layers):
            if i >= self.config.skip_start_idx and i != self.config.num_hidden_layers - 1:
                continue

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                raise NotImplementedError
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

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

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


class DebugLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = DebugModel(config)


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer
    from datasets import load_dataset, load_from_disk

    model_path = "/root/autodl-tmp/models/Meta-Llama-3-8B-Instruct"
    dataset_name = "/root/autodl-tmp/datasets/mix_med_tokenized_v1.5"
    cfg = DebugConfig.from_pretrained(model_path)
    cfg.set_custom_kwargs(skip_start_idx=-1)
    model = DebugLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda:0")
    data = load_from_disk(dataset_name)

    loss_dict = {}
    for skip_layer_idx in range(1, 31):
        sum_loss, cnt = 0, 0
        for i, d in enumerate(data):
            with torch.no_grad():
                model.model.config.skip_start_idx = skip_layer_idx
                ipt_ids = torch.tensor(d["input_ids"], dtype=torch.int64, device="cuda:0")
                out = model(input_ids=ipt_ids, 
                            use_cache=False,
                            labels=ipt_ids)
                sum_loss += out.loss.item()
                cnt += 1
                if cnt == 100:
                    break
        avg_loss = sum_loss / cnt
        print(skip_layer_idx, avg_loss)
    