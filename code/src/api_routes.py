# # Copyright (C) KonaAI - All Rights Reserved
"""Register all API routers"""
from src.automl.api import automl_router
from src.butler.api import butler_router
from src.butler.api_setup import app  # Import app after it's created
from src.insight_agent.api import insight_agent_router
from src.sql_agent.api import sql_agent_router

# Configure main router after all components are initialized
app.include_router(butler_router, prefix="/api")
app.include_router(automl_router, prefix="/api/automl")
app.include_router(insight_agent_router, prefix="/api/insight_agent")
app.include_router(sql_agent_router, prefix="/api/sql_agent")
