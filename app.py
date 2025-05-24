from flask import Flask, request, jsonify
from datetime import datetime, timezone
import logging
from VectorAgent.embedding_service import embed_and_store_in_pinecone  # <- NEW IMPORT
from VectorAgent.qa_service import generate_answer
import markdown

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route('/api/data', methods=['POST'])
def receive_data():
    try:
        if not request.is_json:
            return jsonify({'success': False, 'message': 'Content-Type must be application/json'}), 400

        json_data = request.get_json()

        required_fields = ['timestamp', 'data']
        missing_fields = [field for field in required_fields if field not in json_data]
        if missing_fields:
            return jsonify({'success': False, 'message': f'Missing required fields: {", ".join(missing_fields)}'}), 400

        data_obj = json_data['data']
        if not isinstance(data_obj, dict):
            return jsonify({'success': False, 'message': 'Data field must be an object'}), 400

        required_data_fields = ['userId', 'scrapedTextData', 'url']
        missing_data_fields = [field for field in required_data_fields if field not in data_obj]
        if missing_data_fields:
            return jsonify({'success': False, 'message': f'Missing required data fields: {", ".join(missing_data_fields)}'}), 400

        try:
            datetime.fromisoformat(json_data['timestamp'].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({'success': False, 'message': 'Invalid timestamp format. Use ISO format'}), 400

        url = data_obj['url']
        if not url.startswith(('http://', 'https://')):
            return jsonify({'success': False, 'message': 'Invalid URL format'}), 400

        if not data_obj['userId'] or not isinstance(data_obj['userId'], str):
            return jsonify({'success': False, 'message': 'userId must be a non-empty string'}), 400

        if not data_obj['scrapedTextData'] or not isinstance(data_obj['scrapedTextData'], str):
            return jsonify({'success': False, 'message': 'scrapedTextData must be a non-empty string'}), 400

        logger.info(f"Received data from user {data_obj['userId']}: {len(data_obj['scrapedTextData'])} characters from {data_obj['url']}")

        embed_and_store_in_pinecone(data_obj, json_data['timestamp'])

        return jsonify({'success': True, 'message': 'Data received and stored successfully'}), 201

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/data/user', methods=['POST'])
def get_data_by_user():
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400

        json_data = request.get_json()

        required_fields = ['userId', 'prompt']
        missing_fields = [field for field in required_fields if field not in json_data]
        if missing_fields:
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

        user_id = json_data['userId']
        if not user_id or not isinstance(user_id, str) or len(user_id.strip()) == 0:
            return jsonify({'error': 'userId must be a non-empty string'}), 400

        prompt = json_data['prompt']
        if not prompt or not isinstance(prompt, str) or len(prompt.strip()) == 0:
            return jsonify({'error': 'prompt must be a non-empty string'}), 400

        if len(prompt.strip()) > 1000:
            return jsonify({'error': 'prompt must be less than 1000 characters'}), 400

        logger.info(f"Received request from user {user_id}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")

        #return jsonify({'data': 'dummy data'}), 200
        answer = generate_answer(user_id, prompt)
        
        answer = markdown.markdown(answer)
        return jsonify({'data': answer}), 200


    except Exception as e:
        logger.error(f"Error processing user request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now(timezone.utc).isoformat() + 'Z'}), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
