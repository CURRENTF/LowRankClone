import torch
from transformers.models.llama.modeling_llama import *
from tools.global_state import hyper_params, data_cls_reversed_dict
from accelerate import Accelerator
import wandb


accelerator = Accelerator()


class TinyBertLlamaForCausalLM(LlamaForCausalLM):
    def set_teacher(self, teacher_model: LlamaForCausalLM):
        self.teacher: LlamaForCausalLM = teacher_model
        self.kl_temperature = self.config.kl_temperature
        self.check_data_cls_loss = self.config.check_data_cls_loss
        self.cur_step = 0
        self.student_loss_sum = 0
        self.teacher_loss_sum = 0
        self.kd_loss_sum = 0
        self.mse_loss_sum = 0
        self.attn_mse_loss_sum = 0

        assert len(self.model.layers) == len(self.teacher.model.layers)
        self.num_layers = len(self.model.layers)
        self.align_matrix = nn.ModuleList([
            nn.Linear(self.config.hidden_size, self.teacher.config.hidden_size, bias=False, dtype=torch.bfloat16) 
            for _ in self.model.layers
        ])

        self.post_init()

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
        output_attentions = True
        output_hidden_states = True
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

        #  kd loss
        with torch.no_grad():
            teacher_out = self.teacher(input_ids, labels=labels, output_hidden_states=True, output_attentions=True)
        target_logp = F.log_softmax(logits / self.kl_temperature, dim=-1)
        raw_logp = F.log_softmax(teacher_out.logits / self.kl_temperature, dim=-1)
        kd_loss = F.kl_div(target_logp, raw_logp, log_target=True, reduction="batchmean")

        # attention maploss
        attn_mse_loss = 0
        # print("teacher attn", teacher_out.attentions)
        # print("student attn", outputs.attentions)
        for i in range(self.num_layers):
            teacher_attn_map = torch.where(
                teacher_out.attentions[i] <= 1e-2,
                torch.zeros_like(teacher_out.attentions[i]), teacher_out.attentions[i]
            )
            student_attn_map = torch.where(
                outputs.attentions[i] <= 1e-2,
                torch.zeros_like(outputs.attentions[i]), outputs.attentions[i]
            )
            attn_mse_loss = attn_mse_loss + F.mse_loss(student_attn_map, teacher_attn_map)

        # loss
        mse_loss = 0
        for i in range(self.num_layers):
            mse_loss = F.mse_loss(self.align_matrix[i](outputs.hidden_states[i]), teacher_out.hidden_states[i]) + mse_loss

        # log
        if accelerator.is_main_process:
            loss_log_steps = hyper_params["gradient_accumulation_steps"] * 10
            self.student_loss_sum += loss.item()
            self.teacher_loss_sum += teacher_out.loss.item()
            self.mse_loss_sum += mse_loss.item()
            self.kd_loss_sum += kd_loss.item()
            self.attn_mse_loss_sum += attn_mse_loss.item()

            if (self.cur_step + 1) % loss_log_steps == 0:
                _step = (self.cur_step + 1) // loss_log_steps
                _log_dict = {
                    "target_loss": self.student_loss_sum / loss_log_steps,
                    "raw_loss": self.teacher_loss_sum / loss_log_steps,
                    "logits_loss": self.kd_loss_sum / loss_log_steps,
                    "mse_loss": self.mse_loss_sum / loss_log_steps,
                    "attn_mse_loss_sum": self.attn_mse_loss_sum / loss_log_steps,
                }
                
                wandb.log(_log_dict, step=_step)
                # 
                self.student_loss_sum, self.teacher_loss_sum, self.kd_loss_sum = 0, 0, 0
                self.mse_loss_sum = 0
                self.attn_mse_loss_sum = 0

        self.cur_step += 1

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss + kd_loss + mse_loss + attn_mse_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )