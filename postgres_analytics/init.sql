CREATE DATABASE datauser;

-- Connect to the analytics database to create schemas
\c analytics;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS analytics;

-- Sample table for testing
CREATE TABLE IF NOT EXISTS raw.example_data (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    value NUMERIC,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO raw.example_data (name, value)
VALUES
('item1', 10.5),
('item2', 20.1),
('item3', 15.7),
('item4', 10.5),
('item5', 20.1),
('item6', 15.7);

-- Create read-only user
CREATE USER analytics_reader WITH PASSWORD 'Admin6067';
GRANT USAGE ON SCHEMA raw TO analytics_reader;
GRANT USAGE ON SCHEMA analytics TO analytics_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA raw TO analytics_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO analytics_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA raw GRANT SELECT ON TABLES TO analytics_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT SELECT ON TABLES TO analytics_reader;

-- Create ETL user for Airflow to use
CREATE USER etl_user WITH PASSWORD 'Admin6067';
GRANT USAGE, CREATE ON SCHEMA raw TO etl_user;
GRANT USAGE, CREATE ON SCHEMA analytics TO etl_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA raw TO etl_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA analytics TO etl_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA raw TO etl_user;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA analytics TO etl_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA raw GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO etl_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO etl_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA raw GRANT USAGE ON SEQUENCES TO etl_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA analytics GRANT USAGE ON SEQUENCES TO etl_user;

DROP TABLE IF EXISTS analytics.iris_dataset;

CREATE TABLE analytics.iris_dataset (
    id SERIAL PRIMARY KEY,
    sepal_length FLOAT,
    sepal_width FLOAT,
    petal_length FLOAT,
    petal_width FLOAT,
    species VARCHAR(50)
);

\copy analytics.iris_dataset (sepal_length, sepal_width, petal_length, petal_width, species)
FROM '/tmp/iris_data.csv' WITH CSV HEADER;