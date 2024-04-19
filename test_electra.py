import torch
import torch.nn as nn
import random
import os
import csv
import logging
from torch.utils.data import dataloader

# KoElectra 
from model.dataloader_electra import WellnessTextClassificationDataset
from model.classifier_electra import KoELECTRAforSequenceClassfication
from transformers import ElectraModel, ElectraConfig, ElectraTokenizer


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# 경고 무시
logging.getLogger("transformers").setLevel(logging.ERROR)

def load_wellness_answer():
    root_path = "."
    category_path = f"{root_path}/data/category.txt"
    answer_path = f"{root_path}/data/answer_v2.txt"

    c_f = open(category_path, 'r')
    a_f = open(answer_path, 'r')

    category_lines = c_f.readlines()
    answer_lines = a_f.readlines()

    category = {}
    answer = {}
    for line_num, line_data in enumerate(category_lines):
        data = line_data.split('    ')
        category[data[1][:-1]] = data[0]

    for line_num, line_data in enumerate(answer_lines):
        data = line_data.split('    ')
        keys = answer.keys()
        if (data[0] in keys):
            answer[data[0]] += [data[1][:-1]]
        else:
            answer[data[0]] = [data[1][:-1]]

    return category, answer


def electra_input(tokenizer, str, device=None, max_seq_len=512):
    index_of_words = tokenizer.encode(str)
    token_type_ids = [0] * len(index_of_words)
    attention_mask = [1] * len(index_of_words)

    # Padding Length
    padding_length = max_seq_len - len(index_of_words)

    # Zero Padding
    index_of_words += [0] * padding_length
    token_type_ids += [0] * padding_length
    attention_mask += [0] * padding_length

    data = {
        'input_ids': torch.tensor([index_of_words]).to(device),
        'token_type_ids': torch.tensor([token_type_ids]).to(device),
        'attention_mask': torch.tensor([attention_mask]).to(device),
    }
    return data

if __name__ == "__main__":
    root_path = "."
    checkpoint_path = f"{root_path}/checkpoint"
    
    ### 모델 ###
    save_ckpt_path = f"{checkpoint_path}/model_electra.pth"
    ### 수정 ###


    # 답변과 카테고리 불러오기
    category, answer = load_wellness_answer()

    ctx = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(ctx)

    # 저장한 Checkpoint 불러오기
    checkpoint = torch.load(save_ckpt_path, map_location=device)

    #koElectra_config
    model_config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-generator")

    model = KoELECTRAforSequenceClassfication(model_config, num_labels=432, hidden_dropout_prob=0.1)
    
    model.load_state_dict(checkpoint['model_state_dict'])

    model.to(ctx)
    model.eval()

    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-generator")



    while 1:

        sent1 = input('\nQuestion: ')  # '요즘 기분이 우울한 느낌이에요'
        sent = str(sent1)
        data = electra_input(tokenizer, sent, device, 512)

        if '종료' in sent:
            break

        if '안녕?' in sent:
            print('Answer : 반가워요! 저는 기룡이에요!')
            continue
        if '안녕!' in sent:
            print('Answer : 반가워요! 저는 기룡이에요!')
            continue
        if '안녕' in sent:
            print('Answer : 반가워요! 저는 기룡이에요!')
            continue   

        output = model(**data)

        logit = output[0]
        softmax_logit = torch.softmax(logit, dim=-1)
        softmax_logit = softmax_logit.squeeze()

        max_index = torch.argmax(softmax_logit).item()
        max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()

        if str(max_index) in category:
            answer_category = category[str(max_index)]  # 인덱스에 해당하는 감정 카테고리를 가져옴
            answer_list = answer[category[str(max_index)]]
            answer_len = len(answer_list) - 1
            answer_index = random.randint(0, answer_len)
            # print(f'Answer: {answer_list[answer_index]}, index: {max_index}, softmax_value: {max_index_value}')
            print(f'Answer: {answer_list[answer_index]} \nindex: {answer_category} \nsoftmax_value: {max_index_value}')

            print('-' * 70)
        else:
            # 키가 딕셔너리에 존재하지 않는 경우를 처리합니다.
            print("키가 딕셔너리에 존재하지 않습니다.")

