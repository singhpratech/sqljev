-- sql-jev-laya for BigQuery, as remote functions backed by a sql-jev-laya gateway on Cloud Run.
-- BigQuery batches rows into each call (max_batching_rows); the gateway answers a batch in shared forward passes.
--
--   SELECT * FROM `proj.support.tickets` t WHERE jev.jev(TO_JSON_STRING(t), 'the customer is angry');
--   SELECT jev.jev_choice(TO_JSON_STRING(t), 'which team?', '["billing","technical","sales"]') AS team, COUNT(*)
--   FROM `proj.support.tickets` t GROUP BY team;
--
-- Options/levels are a JSON array in a STRING (remote functions take no ARRAY arguments).
--
-- 1. Deploy the gateway (deploy/cloudrun/README.md), private, e.g. https://sqljev-xxxx.a.run.app
-- 2. bq mk --connection --location=US --connection_type=CLOUD_RESOURCE sqljev_conn
--    and grant the connection's service account roles/run.invoker on the Cloud Run service.
-- 3. Replace PROJECT, US and the endpoint below, then run this script.

CREATE SCHEMA IF NOT EXISTS `PROJECT.jev`;

CREATE OR REPLACE FUNCTION `PROJECT.jev.jev_prob`(row_json STRING, condition STRING) RETURNS FLOAT64
REMOTE WITH CONNECTION `PROJECT.US.sqljev_conn`
OPTIONS (endpoint = 'https://sqljev-xxxx.a.run.app/bigquery', user_defined_context = [("fn", "jev_prob")],
         max_batching_rows = 1000);

CREATE OR REPLACE FUNCTION `PROJECT.jev.jev`(row_json STRING, condition STRING) RETURNS BOOL
REMOTE WITH CONNECTION `PROJECT.US.sqljev_conn`
OPTIONS (endpoint = 'https://sqljev-xxxx.a.run.app/bigquery', user_defined_context = [("fn", "jev")],
         max_batching_rows = 1000);

CREATE OR REPLACE FUNCTION `PROJECT.jev.jev_score`(row_json STRING, question STRING, levels STRING) RETURNS FLOAT64
REMOTE WITH CONNECTION `PROJECT.US.sqljev_conn`
OPTIONS (endpoint = 'https://sqljev-xxxx.a.run.app/bigquery', user_defined_context = [("fn", "jev_score")],
         max_batching_rows = 1000);

CREATE OR REPLACE FUNCTION `PROJECT.jev.jev_score_norm`(row_json STRING, question STRING, levels STRING) RETURNS FLOAT64
REMOTE WITH CONNECTION `PROJECT.US.sqljev_conn`
OPTIONS (endpoint = 'https://sqljev-xxxx.a.run.app/bigquery', user_defined_context = [("fn", "jev_score_norm")],
         max_batching_rows = 1000);

CREATE OR REPLACE FUNCTION `PROJECT.jev.jev_choice`(row_json STRING, question STRING, options STRING) RETURNS STRING
REMOTE WITH CONNECTION `PROJECT.US.sqljev_conn`
OPTIONS (endpoint = 'https://sqljev-xxxx.a.run.app/bigquery', user_defined_context = [("fn", "jev_choice")],
         max_batching_rows = 1000);

CREATE OR REPLACE FUNCTION `PROJECT.jev.jev_confidence`(row_json STRING, question STRING, kind STRING, options STRING) RETURNS FLOAT64
REMOTE WITH CONNECTION `PROJECT.US.sqljev_conn`
OPTIONS (endpoint = 'https://sqljev-xxxx.a.run.app/bigquery', user_defined_context = [("fn", "jev_confidence")],
         max_batching_rows = 1000);

CREATE OR REPLACE FUNCTION `PROJECT.jev.jev_eval`(row_json STRING, question STRING, kind STRING, options STRING) RETURNS JSON
REMOTE WITH CONNECTION `PROJECT.US.sqljev_conn`
OPTIONS (endpoint = 'https://sqljev-xxxx.a.run.app/bigquery', user_defined_context = [("fn", "jev_eval")],
         max_batching_rows = 1000);
