-- sqljev for Amazon Redshift, as Lambda UDFs. Redshift batches rows into each Lambda invocation.
--
--   SELECT * FROM tickets
--   WHERE jev(JSON_SERIALIZE(OBJECT('subject', subject, 'body', body)), 'the customer is angry');
--
-- 1. Deploy the Lambda (deploy/lambda/README.md): handler sqljev.aws_lambda.handler, env
--    SQLJEV_BACKEND=gateway, SQLJEV_API_URL=https://<your gateway>/v1/eval, SQLJEV_GATEWAY_TOKEN=...
-- 2. Create an IAM role Redshift can assume with lambda:InvokeFunction on it, attach it to the cluster.
-- 3. Replace the role ARN below and run this script.
-- Options/levels are a JSON array in a VARCHAR, e.g. '["billing","technical"]'.

CREATE OR REPLACE EXTERNAL FUNCTION jev(varchar(max), varchar(max)) RETURNS boolean STABLE
LAMBDA 'sqljev' IAM_ROLE 'arn:aws:iam::123456789012:role/redshift-sqljev';

CREATE OR REPLACE EXTERNAL FUNCTION jev_prob(varchar(max), varchar(max)) RETURNS float8 STABLE
LAMBDA 'sqljev' IAM_ROLE 'arn:aws:iam::123456789012:role/redshift-sqljev';

CREATE OR REPLACE EXTERNAL FUNCTION jev_score(varchar(max), varchar(max), varchar(max)) RETURNS float8 STABLE
LAMBDA 'sqljev' IAM_ROLE 'arn:aws:iam::123456789012:role/redshift-sqljev';

CREATE OR REPLACE EXTERNAL FUNCTION jev_score_norm(varchar(max), varchar(max), varchar(max)) RETURNS float8 STABLE
LAMBDA 'sqljev' IAM_ROLE 'arn:aws:iam::123456789012:role/redshift-sqljev';

CREATE OR REPLACE EXTERNAL FUNCTION jev_choice(varchar(max), varchar(max), varchar(max)) RETURNS varchar(256) STABLE
LAMBDA 'sqljev' IAM_ROLE 'arn:aws:iam::123456789012:role/redshift-sqljev';

CREATE OR REPLACE EXTERNAL FUNCTION jev_confidence(varchar(max), varchar(max), varchar(max), varchar(max)) RETURNS float8 STABLE
LAMBDA 'sqljev' IAM_ROLE 'arn:aws:iam::123456789012:role/redshift-sqljev';
