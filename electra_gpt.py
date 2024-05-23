import torch
import random
import os
import logging
import openai
import re

# KoElectra
from classifier import KoELECTRAforSequenceClassfication
from transformers import ElectraModel, ElectraConfig, ElectraTokenizer
from kobert_transformers import get_kobert_model

# warning 출력 안되게
logging.getLogger("transformers").setLevel(logging.ERROR)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_wellness_answer(category_path, answer_path):
    c_f = open(category_path, 'r')
    a_f = open(answer_path, 'r')

    category_lines = c_f.readlines()
    answer_lines = a_f.readlines()

    category = {}
    answer = {}
    for line_num, line_data in enumerate(category_lines):
        data = line_data.split('    ')
        if len(data) != 2:
            print(f"Error in category file at line {line_num}: {line_data}")
        category[data[1][:-1]] = data[0]

    for line_num, line_data in enumerate(answer_lines):
        data = line_data.split('    ')
        keys = answer.keys()
        if len(data) != 2:
            print(f"Error in answer file at line {line_num}: {line_data}")
        if (data[0] in keys):
            answer[data[0]] += [data[1][:-1]]
        else:
            answer[data[0]] = [data[1][:-1]]
    return category, answer


def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator")

    model = KoELECTRAforSequenceClassfication(model_config, num_labels=432, hidden_dropout_prob=0.1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    return model, tokenizer, device

def preprocess_input(tokenizer, sent, device, max_seq_len=512):
    index_of_words = tokenizer.encode(sent)
    token_type_ids = [0] * len(index_of_words)
    attention_mask = [1] * len(index_of_words)
    padding_length = max_seq_len - len(index_of_words)
    index_of_words += [0] * padding_length
    token_type_ids += [0] * padding_length
    attention_mask += [0] * padding_length

    data = {
        'input_ids': torch.tensor([index_of_words]).to(device),
        'token_type_ids': torch.tensor([token_type_ids]).to(device),
        'attention_mask': torch.tensor([attention_mask]).to(device),
        }
    return data

def get_answer(category, answer, output, input_sentence):
    softmax_logit = torch.softmax(output[0], dim=-1).squeeze()
    max_index = torch.argmax(softmax_logit).item()
    max_index_value = softmax_logit[torch.argmax(softmax_logit)].item()

    threshold = 0.35

    selected_categories = []
    for i, value in enumerate(softmax_logit):
        if value > threshold:
            if str(i) in category:
                selected_categories.append(category[str(i)])
                print(f"Softmax 값이 threshold({threshold}) 이상인 카테고리: {category[str(i)]}")

    if not selected_categories:
        return "선택된 카테고리가 없습니다. 다시 입력해주세요", None, max_index_value, []

    all_answers = []
    for category_name in selected_categories:
        if category_name in answer:
            all_answers.extend(answer[category_name])

    if not all_answers:
        return "선택된 카테고리에 대한 답변이 없습니다.", None, max_index_value, []

    selected_answer = random.choice(all_answers)

    return selected_answer, selected_categories, max_index_value, all_answers

def gpt(input_sentence, selected_categories):
    openai.api_key = "OPEN_API_KEY"
    MODEL = "gpt-3.5-turbo"

    # 감정 분류 모델에서 예측한 카테고리
    predicted_category = selected_categories

    # 입력받은 사용자 문장
    user_input = input_sentence

    # 프롬프트 설정 (고정값 사용)
    prompt = f"예측한 카테고리는 '{predicted_category}'입니다. 사용자 문장과 예측한 카테고리를 기반으로 도움과 격려가 되는 부드러운 문장으로 심리 상담 챗봇에 쓸 답변을 생성해주세요. 답변은 마침표로 끝나야 합니다."

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input}
    ]

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0.65,
        max_tokens=150,
        n=1,
    )

    # 응답 처리
    original_response = response.choices[0].message.content
    if not original_response.endswith('.'):
        last_sentence = re.split(r'[.!?]', original_response)[-1].strip()
        if last_sentence:
            final_response = original_response + '.'
        else:
            final_response = original_response
    else:
        final_response = original_response

    return final_response


def chat(message):
    root_path = "."
    #answer_path = f"{root_path}/data/new_answer.txt"
    category_path = f"{root_path}/data/category.txt"
    checkpoint_path = f"{root_path}/checkpoint/electra.pth"

    category, answer = load_wellness_answer(category_path, answer_path)
    model, tokenizer, device = load_model(checkpoint_path)

    sent = str(message)

    data = preprocess_input(tokenizer, sent, device, 512)
    output = model(**data)
    answer, category, max_index_value, all_answers = get_answer(category, answer, output, sent)

    chatbot_answer = gpt(sent,category)
    print(chatbot_answer)
  
    return chatbot_answer

