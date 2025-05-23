from flask import Flask, request, jsonify
from datetime import datetime, timezone
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Store received data (in production, you'd use a database)
received_data = []

@app.route('/api/data', methods=['POST'])
def receive_data():
    """
    Endpoint to receive JSON data with timestamp and data fields
    Data should contain: userId, scrapedTextData, and url
    """
    try:
        # Check if request contains JSON
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json'
            }), 400
        
        # Get JSON data from request
        json_data = request.get_json()
        
        # Validate required top-level fields
        required_fields = ['timestamp', 'data']
        missing_fields = [field for field in required_fields if field not in json_data]
        
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Validate data object structure
        data_obj = json_data['data']
        if not isinstance(data_obj, dict):
            return jsonify({
                'error': 'Data field must be an object'
            }), 400
        
        # Validate required fields in data object
        required_data_fields = ['userId', 'scrapedTextData', 'url']
        missing_data_fields = [field for field in required_data_fields if field not in data_obj]
        
        if missing_data_fields:
            return jsonify({
                'error': f'Missing required data fields: {", ".join(missing_data_fields)}'
            }), 400
        
        # Validate timestamp format
        try:
            # Try to parse timestamp to ensure it's valid
            datetime.fromisoformat(json_data['timestamp'].replace('Z', '+00:00'))
        except ValueError:
            return jsonify({
                'error': 'Invalid timestamp format. Use ISO format (e.g., 2023-12-01T10:30:00Z)'
            }), 400
        
        # Validate URL (basic validation)
        url = data_obj['url']
        if not url.startswith(('http://', 'https://')):
            return jsonify({
                'error': 'Invalid URL format. URL must start with http:// or https://'
            }), 400
        
        # Validate userId is not empty
        if not data_obj['userId'] or not isinstance(data_obj['userId'], str):
            return jsonify({
                'error': 'userId must be a non-empty string'
            }), 400
        
        # Validate scrapedTextData is not empty
        if not data_obj['scrapedTextData'] or not isinstance(data_obj['scrapedTextData'], str):
            return jsonify({
                'error': 'scrapedTextData must be a non-empty string'
            }), 400
        
        # Add server timestamp for tracking
        processed_data = {
            'timestamp': json_data['timestamp'],
            'data': {
                'userId': data_obj['userId'],
                'scrapedTextData': data_obj['scrapedTextData'],
                'url': data_obj['url']
            },
            'received_at': datetime.now(timezone.utc).isoformat() + 'Z'
        }
        
        # Store the data
        received_data.append(processed_data)
        
        # Log the received data
        logger.info(f"Received data from user {data_obj['userId']}: {len(data_obj['scrapedTextData'])} characters from {data_obj['url']}")
        
        return jsonify({
            'message': 'Data received successfully',
            'status': 'success',
            'userId': data_obj['userId']
        }), 201
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            'error': 'Internal server error'
        }), 500

@app.route('/api/data', methods=['GET'])
def get_all_data():
    """
    Endpoint to retrieve all received data
    """
    return jsonify({
        'data': received_data,
        'count': len(received_data)
    }), 200

@app.route('/api/data/<int:data_id>', methods=['GET'])
def get_data_by_id(data_id):
    """
    Endpoint to retrieve specific data by ID
    """
    data_item = next((item for item in received_data if item['id'] == data_id), None)
    
    if data_item:
        return jsonify(data_item), 200
    else:
        return jsonify({
            'error': 'Data not found'
        }), 404

@app.route('/api/data/user/<string:user_id>', methods=['GET'])
def get_data_by_user(user_id):
    """
    Endpoint to retrieve data by specific user ID
    """
    user_data = [item for item in received_data if item['data']['userId'] == user_id]
    
    return jsonify({
        'data': user_data,
        'count': len(user_data),
        'userId': user_id
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat() + 'Z'
    }), 200

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'error': 'Method not allowed'
    }), 405

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)