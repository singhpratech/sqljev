-- sqljev for PostgreSQL and compatibles — no extension, no superuser, works on managed Postgres
-- (Amazon RDS / Aurora, Cloud SQL, AlloyDB, Azure Database for PostgreSQL, Supabase, Neon, Crunchy, ...).
--
--   sqljev judge "postgresql://..." --source public.tickets --prob "the customer is angry"   -- judge once
--   SELECT * FROM tickets t WHERE jev.prob(to_jsonb(t), 'the customer is angry') >= 0.5;    -- read, fast
--
-- How it works: managed Postgres cannot run code that calls out, so judging and reading are split.
--   * `sqljev judge` (or anything that can write to jev.answers) sends rows that have no answer yet for the
--     question to Laya in shared forward passes and stores the answers, keyed by (question, row content).
--   * jev.prob / jev.matches / jev.choice / jev.score / jev.eval read jev.answers: an index lookup per row.
--   Row content is hashed from jsonb (canonical key order), so a row edited later is judged again next time.
--
-- Self-hosted Postgres with plpython3u can instead run pg-jev (https://github.com/realZachi/pg-jev) on Laya:
--   SET jev.api_url = 'http://<sqljev gateway>:8765/v1/systemone';   (plus SET jev.api_key = '<gateway token>')

CREATE SCHEMA IF NOT EXISTS jev;

CREATE TABLE IF NOT EXISTS jev.answers (
    question_key bytea       NOT NULL,     -- jev.question_key(question, kind, options)
    row_hash     bytea       NOT NULL,     -- jev.row_hash(to_jsonb(row))
    answer       jsonb       NOT NULL,     -- {"type":"noul","noul":0.93,...}
    judged_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (question_key, row_hash)
);

CREATE OR REPLACE FUNCTION jev.question_key(question text, kind text DEFAULT 'noul', options text[] DEFAULT NULL)
RETURNS bytea LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT sha256(convert_to(kind || chr(30) || question || chr(30) || coalesce(array_to_string(options, chr(31)), ''), 'UTF8'))
$$;

-- NULL columns are dropped before hashing, as they are before judging.
CREATE OR REPLACE FUNCTION jev.row_hash(r jsonb)
RETURNS bytea LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $$
    SELECT sha256(convert_to(jsonb_strip_nulls(r)::text, 'UTF8'))
$$;

-- Full answer, or NULL when the row has not been judged for this question.
CREATE OR REPLACE FUNCTION jev.eval(r jsonb, question text, kind text DEFAULT 'noul', options text[] DEFAULT NULL)
RETURNS jsonb LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT answer FROM jev.answers
    WHERE question_key = jev.question_key(question, kind, options) AND row_hash = jev.row_hash(r)
$$;

CREATE OR REPLACE FUNCTION jev.prob(r jsonb, condition text)
RETURNS float8 LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT (jev.eval(r, condition, 'noul', NULL) ->> 'noul')::float8
$$;

CREATE OR REPLACE FUNCTION jev.matches(r jsonb, condition text, threshold float8 DEFAULT NULL)
RETURNS boolean LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT jev.prob(r, condition) >= coalesce(threshold, nullif(current_setting('jev.threshold', true), '')::float8, 0.5)
$$;

CREATE OR REPLACE FUNCTION jev.choice(r jsonb, question text, options text[])
RETURNS text LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT jev.eval(r, question, 'choice', options) ->> 'choice'
$$;

CREATE OR REPLACE FUNCTION jev.score(r jsonb, question text, levels text[])
RETURNS float8 LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT (jev.eval(r, question, 'score', levels) ->> 'score')::float8
$$;

CREATE OR REPLACE FUNCTION jev.score_norm(r jsonb, question text, levels text[])
RETURNS float8 LANGUAGE sql STABLE PARALLEL SAFE AS $$
    SELECT jev.score(r, question, levels) / greatest(cardinality(levels) - 1, 1)
$$;

-- Set-based reads: every judged row of a question, to join on jev.row_hash(to_jsonb(t)).
CREATE OR REPLACE FUNCTION jev.answers_for(question text, kind text DEFAULT 'noul', options text[] DEFAULT NULL)
RETURNS TABLE (row_hash bytea, prob float8, choice text, score float8, confidence float8, answer jsonb)
LANGUAGE sql STABLE AS $$
    SELECT a.row_hash, (a.answer ->> 'noul')::float8, a.answer ->> 'choice', (a.answer ->> 'score')::float8,
           (a.answer ->> 'confidence')::float8, a.answer
    FROM jev.answers a WHERE a.question_key = jev.question_key(question, kind, options)
$$;

-- Forget answers for one question, or all of them.
CREATE OR REPLACE FUNCTION jev.forget(question text DEFAULT NULL, kind text DEFAULT 'noul', options text[] DEFAULT NULL)
RETURNS bigint LANGUAGE sql VOLATILE AS $$
    WITH d AS (DELETE FROM jev.answers
               WHERE question IS NULL OR question_key = jev.question_key(question, kind, options) RETURNING 1)
    SELECT count(*) FROM d
$$;
