import os
import json
import uuid
import requests
import mimetypes
import base64
import time
import logging
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Config:

    # 模型名称常量 (与前端 model-selector 的 value 对应)
    GEMINI_PRO = 'gemini-2.5-flash'

    # API Endpoints
    GEMINI_API_ENDPOINT = 'https://gemini'

app = Flask(__name__)

# --- 3. 目录设置 ---
CONVERSATIONS_DIR = 'conversations'
TEMP_UPLOADS_DIR = 'temp_uploads' # 用于临时存放上传的图片
if not os.path.exists(CONVERSATIONS_DIR):
    os.makedirs(CONVERSATIONS_DIR)
if not os.path.exists(TEMP_UPLOADS_DIR):
    os.makedirs(TEMP_UPLOADS_DIR)


# --- 4. 辅助函数 ---

def get_conversation_title(history):
    """从第一条用户消息生成标题"""
    for message in history:
        if message['role'] == 'user' and message.get('content'):
            first_line = message['content'].split('\n')[0]
            return (first_line[:47] + '...') if len(first_line) > 50 else first_line
    return "New Chat"

def encode_image_to_base64(image_path):
    """将图片文件编码为 Base64 字符串"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logging.error(f"Error encoding image {image_path}: {e}")
        return None

def get_mime_type(image_path):
    """获取图片的 MIME 类型"""
    mime_type, _ = mimetypes.guess_type(image_path)
    return mime_type or 'application/octet-stream'


# --- 5. 模型调用逻辑 (重构后) ---

def _call_gemini_api(model, prompt, image_paths):
    """调用 Gemini API，支持多图片"""
    logging.info(f"Calling Gemini API with model: {model}")
    data = {'prompt': prompt, 'model': model}
    files_to_upload = []
    opened_files = []

    try:
        for path in image_paths:
            file_object = open(path, 'rb')
            opened_files.append(file_object)
            files_to_upload.append(('images', (os.path.basename(path), file_object, get_mime_type(path))))
        
        response = requests.post(Config.GEMINI_API_ENDPOINT, files=files_to_upload, data=data)
        response.raise_for_status()
        return response.text
    finally:
        # 确保所有打开的文件都被关闭
        for f in opened_files:
            f.close()


def get_model_response(model, prompt, image_paths):
    """根据模型名称调度到对应的API调用函数"""
    if "gemini" in model:
        return _call_gemini_api(model, prompt, image_paths)
    else:
        raise ValueError(f"Unsupported model specified: {model}")


# --- 6. Flask 路由 (API Endpoints) ---

@app.route('/')
def index():
    """渲染主聊天界面"""
    return render_template('index.html')

# get_conversations, get_conversation_history, search_all_conversations 路由保持不变...
@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Lists all saved conversations, sorted by last modification time."""
    conversations = []
    if not os.path.exists(CONVERSATIONS_DIR):
        return jsonify([])

    try:
        file_paths = [
            os.path.join(CONVERSATIONS_DIR, f) 
            for f in os.listdir(CONVERSATIONS_DIR) 
            if f.endswith('.json')
        ]
        sorted_files = sorted(file_paths, key=os.path.getmtime, reverse=True)
        for filepath in sorted_files:
            try:
                filename = os.path.basename(filepath)
                conv_id = filename.split('.')[0]
                with open(filepath, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    title = get_conversation_title(history) 
                    conversations.append({'id': conv_id, 'title': title})
            except (json.JSONDecodeError, IOError, IndexError) as e:
                logging.warning(f"Could not read/parse {filepath}: {e}")
    except Exception as e:
        logging.error(f"Error listing conversations: {e}")
        return jsonify({"error": "Failed to retrieve conversations"}), 500
    return jsonify(conversations)

@app.route('/api/conversation/<conversation_id>', methods=['GET'])
def get_conversation_history(conversation_id):
    """Retrieves the full history for a single conversation."""
    filepath = os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")
    if not os.path.exists(filepath):
        return jsonify({'error': 'Conversation not found'}), 404
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            history = json.load(f)
            return jsonify(history)
    except Exception as e:
        return jsonify({'error': f"Could not load conversation: {e}"}), 500

@app.route('/api/search/all', methods=['GET'])
def search_all_conversations():
    """Searches for a query across all conversation files."""
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    all_results = []
    for filename in os.listdir(CONVERSATIONS_DIR):
        if not filename.endswith('.json'):
            continue
        filepath = os.path.join(CONVERSATIONS_DIR, filename)
        conv_id = filename.split('.')[0]
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                history = json.load(f)
            title = get_conversation_title(history)
            matches = []
            for message in history:
                content = message.get('content', '')
                if query in content.lower():
                    start_index = content.lower().find(query)
                    snippet_start = max(0, start_index - 30)
                    snippet_end = min(len(content), start_index + len(query) + 30)
                    snippet = "..." + content[snippet_start:snippet_end] + "..."
                    matches.append({'role': message['role'], 'snippet': snippet})
            if matches:
                all_results.append({
                    'id': conv_id,
                    'title': title,
                    'matches': matches
                })
        except Exception as e:
            logging.warning(f"Error processing file {filename} for search: {e}")
    return jsonify(all_results)

# --- 主聊天路由---
@app.route('/api/chat', methods=['POST'])
def chat():
    # --- a. 解析请求 ---
    conversation_id = request.form.get('conversation_id')
    prompt = request.form.get('prompt', '')
    model = request.form.get('model', Config.GEMINI_PRO) # 默认模型
    image_files = request.files.getlist('images') # 获取所有名为'images'的文件

    # --- b. 管理会话ID和历史记录 ---
    is_new_conversation = False
    if not conversation_id or conversation_id == 'null':
        conversation_id = str(uuid.uuid4())
        is_new_conversation = True
        conversation_history = []
    else:
        filepath = os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                conversation_history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            conversation_history = []
            
    # --- c. 处理并保存上传的图片 ---
    saved_image_paths = []
    user_content_parts = [prompt]
    try:
        for image_file in image_files:
            if image_file and image_file.filename:
                filename = secure_filename(image_file.filename)
                filepath = os.path.join(TEMP_UPLOADS_DIR, f"{uuid.uuid4()}_{filename}")
                image_file.save(filepath)
                saved_image_paths.append(filepath)
                user_content_parts.append(f"[Image: {filename}]")

        # --- d. 构建用户消息并更新历史 ---
        user_message_content = "\n".join(user_content_parts)
        user_message = {"role": "user", "content": user_message_content}
        conversation_history.append(user_message)
        
        formatted_prompt_for_gemini = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
        logging.info(f"formatted_prompt: {formatted_prompt_for_gemini}")
        # --- e. 调用模型并获取响应 ---
        model_response = get_model_response(model, formatted_prompt_for_gemini, saved_image_paths)
        
        # --- f. 保存AI响应并更新文件 ---
        ai_message = {"role": "assistant", "content": model_response}
        conversation_history.append(ai_message)
        
        conv_filepath = os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")
        with open(conv_filepath, 'w', encoding='utf-8') as f:
            json.dump(conversation_history, f, indent=2, ensure_ascii=False)
            
        # --- g. 构造并返回前端数据 ---
        return_data = {'response': model_response, 'conversation_id': conversation_id}
        if is_new_conversation:
            return_data['new_conversation_created'] = True
        return jsonify(return_data)

    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}", exc_info=True)
        # 如果出错，从历史记录中移除最后添加的用户消息，防止保存不完整的对话
        if conversation_history and conversation_history[-1]['role'] == 'user':
            conversation_history.pop()
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500

    finally:
        # --- h. 清理临时图片文件 ---
        for path in saved_image_paths:
            try:
                os.remove(path)
                logging.info(f"Cleaned up temporary file: {path}")
            except OSError as e:
                logging.error(f"Error cleaning up file {path}: {e}")

if __name__ == '__main__':
    # 建议不要在生产环境中使用 debug=True
    app.run(host='0.0.0.0', port=8810, debug=False)