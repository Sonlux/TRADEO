from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Sample data (similar to what we had in React's mock data)
mock_api_data = {
    "GOOGL": {
        "info": { "name": "Alphabet Inc. (from API)", "price": "$2810.75", "change": "+15.45 (0.55%)" },
        "history": [
            { "date": "2023-02-01", "price": 160 },
            { "date": "2023-02-08", "price": 162 },
            { "date": "2023-02-15", "price": 165 },
            { "date": "2023-02-22", "price": 158 },
            { "date": "2023-03-01", "price": 163 },
        ]
    },
    "AAPL": {
        "info": { "name": "Apple Inc. (from API)", "price": "$177.30", "change": "+1.50 (0.85%)" },
        "history": [
            { "date": "2023-02-01", "price": 170 },
            { "date": "2023-02-08", "price": 172 },
            { "date": "2023-02-15", "price": 165 },
            { "date": "2023-02-22", "price": 168 },
            { "date": "2023-03-01", "price": 175 },
        ]
    }
}

@app.route('/api/stockdata/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    symbol_upper = symbol.upper()
    if symbol_upper in mock_api_data:
        return jsonify(mock_api_data[symbol_upper])
    else:
        return jsonify({"error": "Symbol not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000) # Runs on http://localhost:5000