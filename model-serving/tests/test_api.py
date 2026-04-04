from fastapi.testclient import TestClient

from gemma_serving.app import JobState, create_low_cost_app


def test_health_endpoint_reports_ok() -> None:
    with TestClient(create_low_cost_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["gateway"] == "StubLowCostGateway"


def test_rewrite_job_completes() -> None:
    with TestClient(create_low_cost_app()) as client:
        submit_response = client.post(
            "/jobs/rewrite",
            json={
                "title": "Vintage watch",
                "description": "Steel case with leather strap",
            },
        )
        job_id = submit_response.json()["job_id"]

        for _ in range(10):
            status_response = client.get(f"/jobs/{job_id}")
            payload = status_response.json()
            if payload["status"] == JobState.SUCCEEDED.value:
                break
        else:  # pragma: no cover - defensive polling path
            raise AssertionError("job did not complete")

    assert payload["result"]["title"].startswith("eBay: ")


def test_rewrite_cache_returns_immediate_success() -> None:
    with TestClient(create_low_cost_app()) as client:
        request = {
            "title": "Desk lamp",
            "description": "Brass finish with adjustable arm",
        }
        first_response = client.post("/jobs/rewrite", json=request)
        first_job_id = first_response.json()["job_id"]

        for _ in range(10):
            status_response = client.get(f"/jobs/{first_job_id}")
            if status_response.json()["status"] == JobState.SUCCEEDED.value:
                break

        second_response = client.post("/jobs/rewrite", json=request)

    assert second_response.status_code == 200
    assert second_response.json()["status"] == JobState.SUCCEEDED.value
    assert second_response.json()["cached"] is True