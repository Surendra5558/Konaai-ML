# # Copyright (C) KonaAI - All Rights Reserved
import pytest


# check home page
@pytest.mark.run(order=1)
def test_home_page(client):
    response = client.get("/api/health")
    assert response.status_code == 200, "The base URL did not load successfully."
