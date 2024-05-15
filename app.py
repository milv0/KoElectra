import os
import chatbot as KoElectra
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False #UTF-8 설정하려고 추가한 부분! 이거 없으면 스프링으로 응답 보낼 때 유니코드로 나옴


@app.route('/')
def isRunning():
    return "server is running"

@app.route('/chatbot/bert/<int:chat_id>', methods=['POST']) #chat_id에 맞게 채팅방 생성, 밑에 매개변수에 꼭 있어야 함
def reactKobertChatBot(chat_id):
    
    # POST 요청의 본문을 추출
    message_data = request.json
    
    if message_data and 'messageFromFlutter' in message_data:
        message = message_data['messageFromFlutter']
        
        print(message)
    
    sentence = request.args.get("s")
    if message is None or len(message) == 0 or sentence == '\n':
        return jsonify({
            "response": "듣고 있어요. 더 말씀해주세요~"
        })

    most_similar_sentence = KoElectra.chat(message)
    return jsonify({
        "response": most_similar_sentence,
    })


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))