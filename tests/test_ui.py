"""Simple unit tests for the Flask user interface."""

import os
import pytest

# set test config as config for the app
os.environ["CONFIG_PATH"] = "tests/test_config.yaml"

from EnhancedDatasetSearch.search_platform.main import app as flask_app


@pytest.fixture()
def app(config):
    flask_app.config.update({"TESTING": True})
    yield flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()


def test_home_page(client):
    """Test that home page is available."""
    response = client.get("/")
    assert response.status_code == 200


def test_search_results_page_get(client):
    """Test that search results page is available."""
    response = client.get("/search")
    assert response.status_code == 200


def test_search_results_page_post(client):
    """Test that search results page accepts post requests."""
    response = client.post("/search", json={"query": "example_query"})
    assert response.status_code == 200


def test_chatbot_page(client):
    """Test that chatbot page is available."""
    response = client.get("/chatbot")
    assert response.status_code == 200


def test_dataset_detail_page_redirect(client):
    """Test that dataset detail page is redirected to home page with wrong source parameter."""
    response = client.get("/dataset_detail?source=http://wrong_url")
    assert response.status_code == 302  # redirect to home page


def test_nonexistent_page(client):
    """Test that nonexistent page returns 404 status code."""
    response = client.get("/abc")
    assert response.status_code == 404

