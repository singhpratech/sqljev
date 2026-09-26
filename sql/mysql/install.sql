-- sqljev for MySQL 8+ and MariaDB 10.6+ (also Amazon RDS / Aurora MySQL, Cloud SQL, Azure Database for MySQL).
--
--   sqljev judge "mysql+pymysql://..." --source tickets --prob "the customer is angry"     -- judge once
--   SELECT * FROM tickets t
--   WHERE jev_prob(JSON_OBJECT('id', t.id, 'subject', t.subject, 'body', t.body), 'the customer is angry') >= 0.5;
--
-- How it works: MySQL cannot call out from SQL, so judging and reading are split.
--   * `sqljev judge` sends rows that have no answer yet to Laya in shared forward passes and stores the answers
--     in jev_answers, keyed by (question, row content). It prints the JSON_OBJECT(...) expression it hashed.
--   * jev_prob / jev_matches / jev_choice / jev_score / jev_eval read jev_answers: a primary-key lookup per row.
--   Pass the row exactly as `sqljev judge` printed it: every column, in table order (MySQL sorts JSON keys itself;
--   MariaDB keeps the order you write). Use --columns to judge fewer columns, then pass those.
--   Options/levels are a JSON array string, e.g. '["billing", "technical"]'; whitespace does not matter.
--
-- Run with a client that understands DELIMITER (mysql, mariadb, MySQL Workbench), or run each statement alone.

CREATE TABLE IF NOT EXISTS jev_answers (
    question_key BINARY(32) NOT NULL,
    row_hash     BINARY(32) NOT NULL,
    answer       JSON       NOT NULL,
    judged_at    TIMESTAMP  NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (question_key, row_hash)
);

DROP FUNCTION IF EXISTS jev_question_key;
DROP FUNCTION IF EXISTS jev_row_hash;
DROP FUNCTION IF EXISTS jev_eval;
DROP FUNCTION IF EXISTS jev_prob;
DROP FUNCTION IF EXISTS jev_matches;
DROP FUNCTION IF EXISTS jev_choice;
DROP FUNCTION IF EXISTS jev_score;
DROP FUNCTION IF EXISTS jev_score_norm;

DELIMITER //

CREATE FUNCTION jev_question_key(question TEXT, kind VARCHAR(10), options TEXT) RETURNS BINARY(32)
DETERMINISTIC NO SQL
RETURN UNHEX(SHA2(CONCAT(kind, CHAR(30), question, CHAR(30),
                         COALESCE(REGEXP_REPLACE(options, '[[:space:]]+', ''), '')), 256))//

CREATE FUNCTION jev_row_hash(r TEXT) RETURNS BINARY(32)
DETERMINISTIC NO SQL
RETURN UNHEX(SHA2(r, 256))//

CREATE FUNCTION jev_eval(r JSON, question TEXT, kind VARCHAR(10), options TEXT) RETURNS JSON
READS SQL DATA
RETURN (SELECT answer FROM jev_answers
        WHERE question_key = jev_question_key(question, kind, options) AND row_hash = jev_row_hash(r))//

CREATE FUNCTION jev_prob(r JSON, `condition` TEXT) RETURNS DOUBLE
READS SQL DATA
RETURN JSON_EXTRACT(jev_eval(r, `condition`, 'noul', NULL), '$.noul')//

CREATE FUNCTION jev_matches(r JSON, `condition` TEXT, threshold DOUBLE) RETURNS BOOLEAN
READS SQL DATA
RETURN jev_prob(r, `condition`) >= COALESCE(threshold, 0.5)//

CREATE FUNCTION jev_choice(r JSON, question TEXT, options TEXT) RETURNS VARCHAR(255)
READS SQL DATA
RETURN JSON_UNQUOTE(JSON_EXTRACT(jev_eval(r, question, 'choice', options), '$.choice'))//

CREATE FUNCTION jev_score(r JSON, question TEXT, levels TEXT) RETURNS DOUBLE
READS SQL DATA
RETURN JSON_EXTRACT(jev_eval(r, question, 'score', levels), '$.score')//

CREATE FUNCTION jev_score_norm(r JSON, question TEXT, levels TEXT) RETURNS DOUBLE
READS SQL DATA
RETURN jev_score(r, question, levels) / GREATEST(JSON_LENGTH(levels) - 1, 1)//

DELIMITER ;
