"""Unit tests for the FastAPI serving layer and Redis cache."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.serving.api import create_app
from src.serving.cache import RedisCache

# ---------------------------------------------------------------------------
# RedisCache (uses fakeredis for isolation)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_cache():
    try:
        import fakeredis

        fake_client = fakeredis.FakeRedis(decode_responses=True)
        return RedisCache(client=fake_client, ttl=60)
    except ImportError:
        pytest.skip("fakeredis not installed")


class TestRedisCache:
    def test_set_and_get(self, fake_cache):
        fake_cache.set("test_key", {"value": 42})
        result = fake_cache.get("test_key")
        assert result == {"value": 42}

    def test_miss_returns_none(self, fake_cache):
        result = fake_cache.get("nonexistent_key")
        assert result is None

    def test_hit_rate_tracks_correctly(self, fake_cache):
        fake_cache.set("k1", {"x": 1})
        fake_cache.get("k1")  # hit
        fake_cache.get("k2")  # miss
        assert fake_cache.hit_rate == 0.5

    def test_delete(self, fake_cache):
        fake_cache.set("del_key", {"data": "here"})
        fake_cache.delete("del_key")
        assert fake_cache.get("del_key") is None

    def test_make_key_deterministic(self):
        key1 = RedisCache.make_key("prefix", {"a": 1, "b": 2})
        key2 = RedisCache.make_key("prefix", {"b": 2, "a": 1})
        assert key1 == key2

    def test_make_key_includes_prefix(self):
        key = RedisCache.make_key("churn", {"customer_id": "c123"})
        assert key.startswith("churn:")

    def test_stats_returns_dict(self, fake_cache):
        stats = fake_cache.stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats


# ---------------------------------------------------------------------------
# FastAPI endpoints (no real models loaded → 503 expected for ML routes)
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client(fake_cache):
    app = create_app(cache=fake_cache)
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_metrics_returns_cache_stats(self, api_client):
        response = api_client.get("/metrics")
        assert response.status_code == 200
        body = response.json()
        assert "cache" in body


class TestChurnEndpoint:
    VALID_PAYLOAD = {
        "customer_id": "cust_001",
        "total_events": 150,
        "unique_products": 5,
        "total_revenue": 1200.50,
        "avg_order_value": 45.0,
        "avg_session_sec": 180.0,
        "days_since_last_event": 7,
        "customer_tenure_days": 365,
        "active_days": 30,
        "engagement_score": 0.75,
    }

    def test_returns_503_without_model(self, api_client):
        """No model injected → should return 503."""
        response = api_client.post("/predict/churn", json=self.VALID_PAYLOAD)
        assert response.status_code == 503

    def test_validation_rejects_negative_revenue(self, api_client):
        payload = {**self.VALID_PAYLOAD, "total_revenue": -100}
        response = api_client.post("/predict/churn", json=payload)
        assert response.status_code == 422

    def test_validation_rejects_missing_field(self, api_client):
        payload = {k: v for k, v in self.VALID_PAYLOAD.items() if k != "total_events"}
        response = api_client.post("/predict/churn", json=payload)
        assert response.status_code == 422


class TestDocumentClassificationEndpoint:
    def test_returns_503_without_classifier(self, api_client):
        payload = {"doc_id": "d1", "text": "This is a compliance document for review."}
        response = api_client.post("/classify/document", json=payload)
        assert response.status_code == 503

    def test_validation_rejects_short_text(self, api_client):
        payload = {"doc_id": "d1", "text": "Hi"}
        response = api_client.post("/classify/document", json=payload)
        assert response.status_code == 422


class TestCachedChurnEndpoint:
    """Test that cached predictions are returned without hitting the model."""

    def test_cached_response_served(self, fake_cache):
        # Pre-populate cache
        payload = {
            "customer_id": "cached_cust",
            "total_events": 10,
            "unique_products": 2,
            "total_revenue": 200.0,
            "avg_order_value": 20.0,
            "avg_session_sec": 60.0,
            "days_since_last_event": 5,
            "customer_tenure_days": 100,
            "active_days": 10,
            "engagement_score": 0.5,
        }
        cache_key = RedisCache.make_key("churn", payload)
        fake_cache.set(cache_key, {"prediction": 0, "churn_probability": 0.12})

        app = create_app(cache=fake_cache)
        client = TestClient(app)
        response = client.post("/predict/churn", json=payload)
        # Cache hit → model not needed → should NOT be 503
        assert response.status_code == 200
        body = response.json()
        assert body["cached"] is True
        assert body["churn_probability"] == pytest.approx(0.12)
