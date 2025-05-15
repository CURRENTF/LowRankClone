from transformers.models.llama.modeling_llama import *
from tools.global_state import hyper_params, data_cls_reversed_dict
import wandb


class KDLlamaForCausalLM(LlamaForCausalLM):
    def set_teacher(self, teacher_model: LlamaForCausalLM):
        self.teacher: LlamaForCausalLM = teacher_model
        self.kl_temperature = self.config.kl_temperature
        self.check_data_cls_loss = self.config.check_data_cls_loss
        self.cur_step = 0
        self.student_loss_acc = 0
        self.teacher_loss_acc = 0
        self.kd_loss_acc = 0
        self.data_cls_losses = [0] * 8
        self.data_cls_cnt = [0] * 8

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        data_cls: str = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # kd loss
        with torch.no_grad():
            teacher_out = self.teacher(input_ids, labels=labels)
        target_logp = F.log_softmax(logits / self.kl_temperature, dim=-1)
        raw_logp = F.log_softmax(teacher_out.logits / self.kl_temperature, dim=-1)
        kd_loss = F.kl_div(target_logp, raw_logp, log_target=True, reduction="batchmean")

        loss_log_steps = hyper_params["gradient_accumulation_steps"] * 10
        self.student_loss_acc += loss.item()
        self.teacher_loss_acc += teacher_out.loss.item()
        self.kd_loss_acc += kd_loss.item()

        # loss
        if self.check_data_cls_loss:
            assert hidden_states.shape[0] == 1, "only appliable in bs = 1"
            spec_cls = data_cls[0].item()
            self.data_cls_cnt[spec_cls] += 1
            self.data_cls_losses[spec_cls] += loss.item()

        # log
        if (self.cur_step + 1) % loss_log_steps == 0:
            _step = (self.cur_step + 1) // loss_log_steps
            _log_dict = {
                "target_loss": self.student_loss_acc / loss_log_steps,
                "raw_loss": self.teacher_loss_acc / loss_log_steps,
                "logits_loss": self.kd_loss_acc / loss_log_steps,
            }
            if self.check_data_cls_loss:
                _log_dict.update({
                    f"{data_cls_reversed_dict[i]}_loss": _loss / self.data_cls_cnt[i] 
                    for i, _loss in enumerate(self.data_cls_losses) if self.data_cls_cnt[i] > 0
                })
            
            wandb.log(_log_dict, step=_step)
            # 
            self.student_loss_acc, self.teacher_loss_acc, self.kd_loss_acc = 0, 0, 0
            self.data_cls_cnt = [0] * 8
            self.data_cls_losses = [0] * 8

        self.cur_step += 1

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss + kd_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )