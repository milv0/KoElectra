import torch
import random
import os
import logging

# KoElectra 
from model.chatbot.kobert.classifier import KoELECTRAforSequenceClassfication
from transformers import ElectraModel, ElectraConfig, ElectraTokenizer
from kobert_transformers import get_kobert_model

# warning 출력 안되게
logging.getLogger("transformers").setLevel(logging.ERROR)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_wellness_answer(category_path, answer_path):
    c_f = open(category_path, 'r', encoding="UTF8")
    a_f = open(answer_path, 'r', encoding="UTF8")

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
    model.load_state_dict(checkpoint['model_state_dict'], strict = False)
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


def find_most_similar_sentence(input_sentence, candidate_sentences, output):
    model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    def get_sentence_embedding(sentence):
        inputs = tokenizer.encode_plus(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1)
        return sentence_embedding

    input_embedding = get_sentence_embedding(input_sentence)

    selected_sentence = None

    similarities = []

    for candidate in candidate_sentences:
        candidate_embedding = get_sentence_embedding(candidate)
        similarity = torch.cosine_similarity(input_embedding, candidate_embedding, dim=1)
        similarities.append((candidate, similarity.item()))

    top_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

    selected_sentence, _ = random.choice(top_similarities)

    print("유사도 상위 5개 문장과 유사도 수치:")
    for sentence, similarity in top_similarities:
        print(f"유사도: {similarity:.4f} , 문장: {sentence}" )

    return selected_sentence

def chat(message):
    root_path = "."
    answer_path = f"{root_path}/data/answer_R_v1.txt"
    category_path = f"{root_path}/data/category_R.txt"
    checkpoint_path = f"{root_path}/checkpoint/electra_R_v1.pth"

    category, answer = load_wellness_answer(category_path, answer_path)
    model, tokenizer, device = load_model(checkpoint_path)

    sent = str(message)
    if '안녕?' in sent or '안녕!' in sent or '안녕' in sent:
        most_similar_sentence = '반가워요! 저는 기룡이에요!'
        return most_similar_sentence

    data = preprocess_input(tokenizer, sent, device, 512)
    output = model(**data)
    answer, category, max_index_value, all_answers = get_answer(category, answer, output, sent)

    most_similar_sentence = find_most_similar_sentence(sent, all_answers, output)
    return most_similar_sentence