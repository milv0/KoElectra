import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from transformers import ElectraModel, ElectraConfig, ElectraTokenizer

class KoELECTRAforSequenceClassfication(nn.Module):
    def __init__(self, config, num_labels=432, hidden_dropout_prob=0.1):
        super().__init__()
        self.num_labels = num_labels
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.additional_layer_1 = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.additional_layer_2 = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        pooled_output = outputs.last_hidden_state[:, 0, :]  # 변경
        pooled_output = self.pooler(pooled_output)  # 변경

        pooled_output = self.dropout(pooled_output)
        pooled_output = self.additional_layer_1(pooled_output)
        pooled_output = nn.functional.relu(pooled_output)
        pooled_output = self.additional_layer_2(pooled_output)

        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs

def koelectra_input(tokenizer, str, device=None, max_seq_len=512):
    encoding = tokenizer.encode_plus(
        str,
        max_length=max_seq_len,
        pad_to_max_length=True,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    data = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask,
    }

    return data


# import torch
# import torch.nn as nn
# from torch.nn import CrossEntropyLoss, MSELoss
# import torch.nn.functional as F
# from transformers import ElectraModel, ElectraConfig, ElectraTokenizer

# class KoELECTRAforSequenceClassfication(nn.Module):
#     def __init__(self, config, num_labels=432, hidden_dropout_prob=0.1):
#         super().__init__()
#         self.num_labels = num_labels
#         self.electra = ElectraModel(config)
#         self.dropout = nn.Dropout(hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         self.additional_layer_1 = nn.Linear(config.hidden_size, config.hidden_size * 4)
#         self.additional_layer_2 = nn.Linear(config.hidden_size * 4, config.hidden_size)
#         self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
#         self.layer_norm = nn.LayerNorm(config.hidden_size)  # 레이어 정규화 추가

#     def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
#         outputs = self.electra(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             output_hidden_states=True,  # 모든 은닉 상태 출력
#             output_attentions=True,  # Attention 가중치 출력
#         )

#         # 모든 레이어의 출력을 결합
#         hidden_states = torch.stack(outputs.hidden_states, dim=0)
#         pooled_output = torch.mean(hidden_states, dim=0)

#         # Attention 가중치 활용
#         if outputs.attentions is not None:  # attentions가 None이 아닌 경우에만 실행
#             attention_scores = outputs.attentions[-1]  # 마지막 레이어의 Attention 가중치
#             attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]

#             # attention_weights와 pooled_output의 차원 조정
#             pooled_output = pooled_output.permute(1, 0, 2)  # [seq_len, batch_size, hidden_size]
#             attention_weights = attention_weights.permute(2, 0, 1, 3)  # [seq_len, batch_size, num_heads, seq_len]

#             pooled_output = torch.bmm(attention_weights, pooled_output)  # [seq_len, batch_size, hidden_size]
#             pooled_output = pooled_output.permute(1, 0, 2)  # [batch_size, seq_len, hidden_size]

#         pooled_output = self.layer_norm(pooled_output)  # 레이어 정규화 적용
#         pooled_output = self.dropout(pooled_output)
#         pooled_output = self.additional_layer_1(pooled_output)
#         pooled_output = F.gelu(pooled_output)  # 활성화 함수 변경
#         pooled_output = self.additional_layer_2(pooled_output)

#         logits = self.classifier(pooled_output)

#         outputs = (logits,) + outputs[2:]

#         if labels is not None:
#             loss_fct = LabelSmoothingLoss()  # 손실 함수 변경
#             loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs = (loss,) + outputs

#         return outputs
