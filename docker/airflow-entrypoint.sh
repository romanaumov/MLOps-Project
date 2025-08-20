#!/bin/bash
set -e

# Function to wait for database
wait_for_db() {
    echo "Waiting for database..."
    while ! airflow db check; do
        echo "Database not ready, waiting..."
        sleep 5
    done
    echo "Database is ready!"
}

# Initialize database if running webserver and it's the first time
if [ "$1" = "webserver" ]; then
    wait_for_db
    
    # Initialize database (idempotent)
    airflow db init
    
    # Create admin user (if not exists)
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin || echo "User already exists"
fi

# For scheduler and worker, just wait for DB
if [ "$1" = "scheduler" ] || [ "$1" = "celery" ]; then
    wait_for_db
fi

# Execute the original command
exec airflow "$@"