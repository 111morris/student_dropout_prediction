"""
API Endpoint Tests.
Tests all CRUD and prediction endpoints using FastAPI's TestClient.
"""

import pytest


class TestHealthAndRoot:
    """Test system endpoints."""

    def test_root_endpoint(self, client):
        """Root endpoint should return welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_health_endpoint(self, client):
        """Health endpoint should confirm model and DB are ready."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["database_connected"] is True


class TestPredictEndpoint:
    """Test POST /api/v1/predict."""

    def test_predict_returns_valid_response(self, client, sample_student_data):
        """Prediction should return risk_score, risk_status, recommendation."""
        response = client.post("/api/v1/predict", json=sample_student_data)
        assert response.status_code == 200
        data = response.json()

        assert "risk_score" in data
        assert "risk_status" in data
        assert "recommendation" in data

        assert 0.0 <= data["risk_score"] <= 1.0
        assert data["risk_status"] in ["🟢 Low", "🟡 Medium", "🔴 High"]
        assert len(data["recommendation"]) > 0

    def test_predict_high_risk_student(self, client, high_risk_student_data):
        """A high-risk profile should produce a higher risk score."""
        response = client.post("/api/v1/predict", json=high_risk_student_data)
        assert response.status_code == 200
        data = response.json()

        # High risk student should have elevated risk score
        assert data["risk_score"] > 0.0
        assert data["risk_status"] in ["🟡 Medium", "🔴 High"]

    def test_predict_invalid_gpa(self, client, sample_student_data):
        """GPA > 4.0 should return 422 validation error."""
        sample_student_data["gpa"] = 5.0
        response = client.post("/api/v1/predict", json=sample_student_data)
        assert response.status_code == 422

    def test_predict_invalid_gender(self, client, sample_student_data):
        """Invalid gender value should return 422."""
        sample_student_data["gender"] = "Other"
        response = client.post("/api/v1/predict", json=sample_student_data)
        assert response.status_code == 422

    def test_predict_missing_field(self, client, sample_student_data):
        """Missing required field should return 422."""
        del sample_student_data["gpa"]
        response = client.post("/api/v1/predict", json=sample_student_data)
        assert response.status_code == 422

    def test_predict_invalid_department(self, client, sample_student_data):
        """Invalid department should return 422."""
        sample_student_data["department"] = "Philosophy"
        response = client.post("/api/v1/predict", json=sample_student_data)
        assert response.status_code == 422

    def test_predict_negative_attendance(self, client, sample_student_data):
        """Negative attendance rate should return 422."""
        sample_student_data["attendance_rate"] = -5.0
        response = client.post("/api/v1/predict", json=sample_student_data)
        assert response.status_code == 422


class TestStudentsEndpoints:
    """Test CRUD endpoints under /api/v1/students."""

    def test_create_student(self, client, sample_student_data):
        """POST /students should create a student with risk prediction."""
        response = client.post("/api/v1/students", json=sample_student_data)
        assert response.status_code == 201
        data = response.json()

        # Should have all input fields
        assert data["gpa"] == sample_student_data["gpa"]
        assert data["department"] == sample_student_data["department"]

        # Should have prediction results
        assert data["risk_score"] is not None
        assert data["risk_status"] is not None
        assert data["recommendation"] is not None

        # Should have generated ID and timestamps
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_get_student_by_id(self, client, sample_student_data):
        """GET /students/{id} should return the correct student."""
        # Create first
        create_resp = client.post("/api/v1/students", json=sample_student_data)
        student_id = create_resp.json()["id"]

        # Retrieve
        response = client.get(f"/api/v1/students/{student_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == student_id
        assert data["gpa"] == sample_student_data["gpa"]

    def test_get_student_not_found(self, client):
        """GET /students/{invalid_id} should return 404."""
        response = client.get("/api/v1/students/nonexistent-id")
        assert response.status_code == 404

    def test_list_students_empty(self, client):
        """GET /students should return empty list when DB is empty."""
        response = client.get("/api/v1/students")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["students"] == []

    def test_list_students_with_data(self, client, sample_student_data):
        """GET /students should return created students."""
        # Create two students
        client.post("/api/v1/students", json=sample_student_data)
        client.post("/api/v1/students", json=sample_student_data)

        response = client.get("/api/v1/students")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["students"]) == 2

    def test_list_students_pagination(self, client, sample_student_data):
        """GET /students with skip/limit should paginate."""
        # Create 3 students
        for _ in range(3):
            client.post("/api/v1/students", json=sample_student_data)

        # Get first 2
        response = client.get("/api/v1/students?skip=0&limit=2")
        data = response.json()
        assert data["total"] == 3
        assert len(data["students"]) == 2

        # Get remaining
        response = client.get("/api/v1/students?skip=2&limit=2")
        data = response.json()
        assert len(data["students"]) == 1

    def test_delete_student(self, client, sample_student_data):
        """DELETE /students/{id} should remove the student."""
        # Create
        create_resp = client.post("/api/v1/students", json=sample_student_data)
        student_id = create_resp.json()["id"]

        # Delete
        response = client.delete(f"/api/v1/students/{student_id}")
        assert response.status_code == 204

        # Verify gone
        response = client.get(f"/api/v1/students/{student_id}")
        assert response.status_code == 404

    def test_delete_student_not_found(self, client):
        """DELETE /students/{invalid_id} should return 404."""
        response = client.delete("/api/v1/students/nonexistent-id")
        assert response.status_code == 404
