"""
Endpoint: GET /api/models
Liste les modeles disponibles
"""

from http.server import BaseHTTPRequestHandler
import json
import os

# Cle API
API_KEY = os.environ.get("API_KEY", "ATTRITION_SECRET_KEY_2024")


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Verifier la cle API
        api_key = self.headers.get('X-API-Key', '')
        
        if api_key != API_KEY:
            self.send_response(401)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Cle API invalide"}).encode())
            return
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "models": ["rf", "logreg", "nb"],
            "default": "rf",
            "descriptions": {
                "rf": "Random Forest - Bon equilibre precision/rappel",
                "logreg": "Regression Logistique - Simple et interpretable",
                "nb": "Naive Bayes - Rapide mais moins precis"
            }
        }
        
        self.wfile.write(json.dumps(response).encode())
        return
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-API-Key, Content-Type')
        self.end_headers()
        return

