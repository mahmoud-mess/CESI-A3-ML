"""
Endpoint: GET /api
Health check et page d'accueil
"""

from http.server import BaseHTTPRequestHandler
import json


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = {
            "message": "API Attrition Employes - Ready!",
            "version": "1.0.0",
            "endpoints": [
                "/api - Health check",
                "/api/models - Liste des modeles",
                "/api/predict - Prediction d'attrition (POST)"
            ],
            "status": "ok"
        }
        
        self.wfile.write(json.dumps(response).encode())
        return

