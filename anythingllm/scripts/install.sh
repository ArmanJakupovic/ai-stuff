#!/bin/bash

# Set the storage location
export STORAGE_LOCATION=$HOME/anythingllm

# Create the storage directory if it doesn't exist
mkdir -p $STORAGE_LOCATION

# Create the .env file if it doesn't exist
touch "$STORAGE_LOCATION/.env"

# Run the Docker container
docker run -d -p 3001:3001 \
  --cap-add SYS_ADMIN \
  -v ${STORAGE_LOCATION}:/app/server/storage \
  -v ${STORAGE_LOCATION}/.env:/app/server/.env \
  -e STORAGE_DIR="/app/server/storage" \
  mintplexlabs/anythingllm
