from fastapi.testclient import TestClient
from app.main import app
from app.config import settings

client = TestClient(app)

def test_analyze_success():
    csv_content = "department,revenue,orders\nA,100,2\nB,200,4\n"
    response = client.post(
        "/api/v1/analyze",
        data={"question": "What is the average revenue?"},
        files={"file": ("sales.csv", csv_content, "text/csv")}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is the average revenue?"
    assert "dataset_summary" in data

    summary = data["dataset_summary"]
    assert summary["row_count"] == 2
    assert summary["column_count"] == 3
    assert "revenue" in summary["numeric_columns"]
    assert "department" in summary["categorical_columns"]

def test_analyze_invalid_extension():
    response = client.post(
        "/api/v1/analyze",
        data={"question": "Check departments"},
        files={"file": ("sales.xlsx", "fake excel bytes", "application/vnd.ms-excel")}
    )

    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "FileSafetyError"
    assert "Only CSV files are supported" in data["detail"]

def test_analyze_empty_file():
    response = client.post(
        "/api/v1/analyze",
        data={"question": "Check counts"},
        files={"file": ("sales.csv", "", "text/csv")}
    )

    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "FileSafetyError"
    assert "file is empty" in data["detail"].lower()

def test_analyze_missing_question():
    response = client.post(
        "/api/v1/analyze",
        files={"file": ("sales.csv", "a,b\n1,2", "text/csv")}
    )
    # FastAPI returns 422 if a required Form parameter is missing
    assert response.status_code == 422

def test_analyze_blank_question():
    response = client.post(
        "/api/v1/analyze",
        data={"question": "    "},
        files={"file": ("sales.csv", "a,b\n1,2", "text/csv")}
    )

    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "DatasetValidationError"
    assert "Question cannot be empty" in data["detail"]

def test_analyze_row_limit():
    old_max = settings.MAX_DATASET_ROWS
    settings.MAX_DATASET_ROWS = 2
    try:
        csv_content = "a,b\n1,2\n3,4\n5,6"
        response = client.post(
            "/api/v1/analyze",
            data={"question": "analyze rows"},
            files={"file": ("sales.csv", csv_content, "text/csv")}
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "DatasetValidationError"
        assert "exceeds the maximum limit of 2" in data["detail"]
    finally:
        settings.MAX_DATASET_ROWS = old_max

def test_analyze_column_limit():
    old_max = settings.MAX_DATASET_COLUMNS
    settings.MAX_DATASET_COLUMNS = 2
    try:
        csv_content = "a,b,c\n1,2,3\n4,5,6"
        response = client.post(
            "/api/v1/analyze",
            data={"question": "analyze columns"},
            files={"file": ("sales.csv", csv_content, "text/csv")}
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "DatasetValidationError"
        assert "exceeds the maximum limit of 2" in data["detail"]
    finally:
        settings.MAX_DATASET_COLUMNS = old_max

def test_analyze_oversized_file():
    old_max = settings.MAX_UPLOAD_SIZE_MB
    # Set limit to 0MB, so any non-empty file is too large
    settings.MAX_UPLOAD_SIZE_MB = 0
    try:
        csv_content = "a,b\n1,2"
        response = client.post(
            "/api/v1/analyze",
            data={"question": "analyze large file"},
            files={"file": ("sales.csv", csv_content, "text/csv")}
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "FileSafetyError"
        assert "File size exceeds the limit" in data["detail"]
    finally:
        settings.MAX_UPLOAD_SIZE_MB = old_max
