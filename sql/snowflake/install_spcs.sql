-- sql-jev-laya for Snowflake with Laya running INSIDE Snowflake (Snowpark Container Services, GPU).
-- Rows never leave your Snowflake account, and Snowflake batches rows into each service-function call.
--
--   SELECT * FROM tickets t WHERE jev(OBJECT_CONSTRUCT(t.*), 'the customer is angry');
--
-- 1. Build and push the gateway image with the checkpoint baked in (no egress needed at runtime):
--      docker build -f deploy/Dockerfile --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 \
--                   -t <org>-<account>.registry.snowflakecomputing.com/sqljev/public/sqljev_repo/sql-jev-laya:0.1.0 .
--      docker push  <same tag>
--    To serve a fine-tuned checkpoint, bake it in with --build-arg MODEL=<hub id> and set SQLJEV_MODEL below.
-- 2. Run this script.

USE ROLE ACCOUNTADMIN;
CREATE DATABASE IF NOT EXISTS SQLJEV;
CREATE SCHEMA IF NOT EXISTS SQLJEV.PUBLIC;
USE SCHEMA SQLJEV.PUBLIC;
CREATE IMAGE REPOSITORY IF NOT EXISTS sqljev_repo;

CREATE COMPUTE POOL IF NOT EXISTS sqljev_gpu
  MIN_NODES = 1 MAX_NODES = 1
  INSTANCE_FAMILY = GPU_NV_S          -- one A10G; CPU_X64_M works too, ~10x slower
  AUTO_SUSPEND_SECS = 600;

CREATE SERVICE IF NOT EXISTS sqljev_service
  IN COMPUTE POOL sqljev_gpu
  FROM SPECIFICATION $$
spec:
  containers:
  - name: gateway
    image: /sqljev/public/sqljev_repo/sql-jev-laya:0.1.0
    env:
      SQLJEV_BACKEND: local
      SQLJEV_DEVICE: cuda
      SQLJEV_BATCH_SIZE: "128"
      HF_HUB_OFFLINE: "1"
    resources:
      requests: {nvidia.com/gpu: 1}
      limits: {nvidia.com/gpu: 1}
    readinessProbe:
      port: 8765
      path: /health
  endpoints:
  - name: api
    port: 8765
$$;

-- Service functions: Snowflake POSTs {"data": [[rownum, args...], ...]} batches to the gateway.
CREATE OR REPLACE FUNCTION jev_prob(row VARIANT, condition STRING) RETURNS FLOAT
  SERVICE = sqljev_service ENDPOINT = api MAX_BATCH_ROWS = 1000 AS '/snowflake/jev_prob';
CREATE OR REPLACE FUNCTION jev(row VARIANT, condition STRING) RETURNS BOOLEAN
  SERVICE = sqljev_service ENDPOINT = api MAX_BATCH_ROWS = 1000 AS '/snowflake/jev';
CREATE OR REPLACE FUNCTION jev(row VARIANT, condition STRING, threshold FLOAT) RETURNS BOOLEAN
  SERVICE = sqljev_service ENDPOINT = api MAX_BATCH_ROWS = 1000 AS '/snowflake/jev';
CREATE OR REPLACE FUNCTION jev_score(row VARIANT, question STRING, levels ARRAY) RETURNS FLOAT
  SERVICE = sqljev_service ENDPOINT = api MAX_BATCH_ROWS = 1000 AS '/snowflake/jev_score';
CREATE OR REPLACE FUNCTION jev_score_norm(row VARIANT, question STRING, levels ARRAY) RETURNS FLOAT
  SERVICE = sqljev_service ENDPOINT = api MAX_BATCH_ROWS = 1000 AS '/snowflake/jev_score_norm';
CREATE OR REPLACE FUNCTION jev_choice(row VARIANT, question STRING, options ARRAY) RETURNS STRING
  SERVICE = sqljev_service ENDPOINT = api MAX_BATCH_ROWS = 1000 AS '/snowflake/jev_choice';
CREATE OR REPLACE FUNCTION jev_confidence(row VARIANT, question STRING, kind STRING, options ARRAY) RETURNS FLOAT
  SERVICE = sqljev_service ENDPOINT = api MAX_BATCH_ROWS = 1000 AS '/snowflake/jev_confidence';
CREATE OR REPLACE FUNCTION jev_eval(row VARIANT, question STRING, kind STRING, options ARRAY) RETURNS VARIANT
  SERVICE = sqljev_service ENDPOINT = api MAX_BATCH_ROWS = 1000 AS '/snowflake/jev_eval';

-- Grant to analysts:
--   GRANT USAGE ON DATABASE SQLJEV TO ROLE analyst; GRANT USAGE ON SCHEMA SQLJEV.PUBLIC TO ROLE analyst;
--   GRANT SERVICE ROLE sqljev_service!ALL_ENDPOINTS_USAGE TO ROLE analyst;
--   GRANT USAGE ON ALL FUNCTIONS IN SCHEMA SQLJEV.PUBLIC TO ROLE analyst;
