import sys
from pathlib import Path

# Explicitly tell Python where the root of your project is
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from fastapi.testclient import TestClient
from api.main import app

def test_root_endpoint():
    """Test if the API is alive and reachable"""
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()

def test_health_check():
    """Test the health check endpoint"""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        
        # Check specific keys instead of an exact dictionary match
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data