<!-- To run locally for testing -->
# PowerShell or CMD (Compose v2)
docker compose -f docker/docker-compose.yml -f docker/docker-compose.local.yml up -d

# In Azure Cloud
docker compose --env-file .env.azure -f docker-compose.yml -f docker-compose.azure.yml up -d
