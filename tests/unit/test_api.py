"""
Unit tests for API endpoints.

Tests the FastAPI routes for design creation, retrieval, and management.
"""

import pytest
from fastapi import status


def test_root_endpoint(client):
    """Test the root endpoint returns correct information."""
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["message"] == "stuff-made-easy: GenAI PCB Design Platform"
    assert data["version"] == "0.1.0"
    assert "features" in data


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "status" in data
    assert "database" in data


def test_create_design(client, sample_design_request):
    """Test creating a new design project."""
    response = client.post("/api/v1/designs", json=sample_design_request)
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["name"] == sample_design_request["name"]
    assert data["status"] == "draft"
    assert "id" in data


def test_list_designs_empty(client):
    """Test listing designs when database is empty."""
    response = client.get("/api/v1/designs")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


def test_list_designs_with_data(client, sample_design_request):
    """Test listing designs after creating one."""
    # Create a design
    create_response = client.post("/api/v1/designs", json=sample_design_request)
    assert create_response.status_code == status.HTTP_201_CREATED
    
    # List designs
    list_response = client.get("/api/v1/designs")
    assert list_response.status_code == status.HTTP_200_OK
    data = list_response.json()
    assert len(data) == 1
    assert data[0]["name"] == sample_design_request["name"]


def test_get_design_by_id(client, sample_design_request):
    """Test retrieving a specific design by ID."""
    # Create a design
    create_response = client.post("/api/v1/designs", json=sample_design_request)
    design_id = create_response.json()["id"]
    
    # Get the design
    get_response = client.get(f"/api/v1/designs/{design_id}")
    assert get_response.status_code == status.HTTP_200_OK
    data = get_response.json()
    assert data["id"] == design_id
    assert data["name"] == sample_design_request["name"]


def test_get_nonexistent_design(client):
    """Test retrieving a design that doesn't exist."""
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/api/v1/designs/{fake_id}")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_delete_design(client, sample_design_request):
    """Test deleting a design project."""
    # Create a design
    create_response = client.post("/api/v1/designs", json=sample_design_request)
    design_id = create_response.json()["id"]
    
    # Delete the design
    delete_response = client.delete(f"/api/v1/designs/{design_id}")
    assert delete_response.status_code == status.HTTP_204_NO_CONTENT
    
    # Verify it's gone
    get_response = client.get(f"/api/v1/designs/{design_id}")
    assert get_response.status_code == status.HTTP_404_NOT_FOUND


def test_create_design_invalid_prompt_too_short(client):
    """Test creating a design with a prompt that's too short."""
    invalid_request = {
        "name": "Test",
        "prompt": "short"  # Less than 10 characters
    }
    response = client.post("/api/v1/designs", json=invalid_request)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_create_design_missing_name(client):
    """Test creating a design without a name."""
    invalid_request = {
        "prompt": "Design a simple LED circuit"
    }
    response = client.post("/api/v1/designs", json=invalid_request)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY