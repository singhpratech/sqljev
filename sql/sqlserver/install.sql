-- sqljev for SQL Server / Azure SQL — plain-language questions over your rows.
--
--   EXEC jev.judge N'dbo.tickets', N'the customer is angry';              -- judge every row once
--   SELECT * FROM dbo.tickets AS t
--   WHERE jev.prob((SELECT t.* FOR JSON PATH, WITHOUT_ARRAY_WRAPPER), N'the customer is angry') >= 0.5;
--
-- How it works: T-SQL scalar functions cannot call out, so judging and reading are split.
--   * jev.judge serialises each row with FOR JSON, skips rows already answered (by content hash), and sends the
--     rest in batches to a sqljev gateway, which runs Laya and answers a whole batch in shared forward passes.
--     Answers land in jev.answers, keyed by (question, row content).
--   * jev.prob / jev.matches / jev.choice / jev.score / jev.eval read jev.answers: an index seek per row.
--   Re-running a question, changing the threshold or sorting by probability is free; a changed row is re-judged.
--
-- Calling the gateway from T-SQL needs sp_invoke_external_rest_endpoint (SQL Server 2025, Azure SQL Database,
-- Azure SQL Managed Instance) and the gateway on https:// port 443 under a DNS name with a publicly trusted
-- certificate (Azure SQL Database additionally only allows Azure-hosted endpoints, e.g. *.azurewebsites.net,
-- *.azurecontainerapps.io). On SQL Server 2016-2022, or without outbound HTTPS, fill
-- the same table from outside:  sqljev judge-sqlserver "mssql+pyodbc://..." --source dbo.tickets --prob "..."
--
-- Setup:
--   sqljev gateway --host 0.0.0.0 --port 8443 --certfile cert.pem --keyfile key.pem   (SQLJEV_GATEWAY_TOKEN=...)
--   EXEC sp_configure 'external rest endpoint enabled', 1; RECONFIGURE;                  (SQL Server 2025)
--   CREATE DATABASE SCOPED CREDENTIAL [https://gateway.example.com]
--     WITH IDENTITY = 'HTTPEndpointHeaders', SECRET = '{"X-Jev-Token":"<token>"}';      (needs a master key)
--   EXEC jev.configure @gateway_url = N'https://gateway.example.com/sqlserver',
--                      @credential = N'https://gateway.example.com';

IF SCHEMA_ID(N'jev') IS NULL EXEC (N'CREATE SCHEMA jev');
GO

IF OBJECT_ID(N'jev.settings', N'U') IS NULL
CREATE TABLE jev.settings (
    name  sysname        NOT NULL CONSTRAINT PK_jev_settings PRIMARY KEY,
    value nvarchar(4000) NOT NULL
);
GO

IF OBJECT_ID(N'jev.answers', N'U') IS NULL
CREATE TABLE jev.answers (
    question_key binary(32)    NOT NULL,     -- jev.question_key(question, kind, options)
    row_hash     binary(32)    NOT NULL,     -- jev.row_hash(FOR JSON text of the row)
    answer       nvarchar(max) NOT NULL,     -- {"type":"noul","noul":0.93,...}
    judged_at    datetime2(0)  NOT NULL CONSTRAINT DF_jev_answers_judged_at DEFAULT sysutcdatetime(),
    CONSTRAINT PK_jev_answers PRIMARY KEY (question_key, row_hash)
);
GO

-- Options are normalised (JSON array -> unit-separated list) so '["a", "b"]' and '["a","b"]' share answers.
CREATE OR ALTER FUNCTION jev.question_key (@question nvarchar(4000), @kind varchar(10), @options nvarchar(max))
RETURNS binary(32)
AS
BEGIN
    DECLARE @opts nvarchar(max) = N'';
    IF @options IS NOT NULL AND ISJSON(@options) = 1
        SELECT @opts = STRING_AGG(CAST(value AS nvarchar(max)), NCHAR(31)) WITHIN GROUP (ORDER BY CAST([key] AS int))
        FROM OPENJSON(@options);
    RETURN HASHBYTES('SHA2_256', CONCAT(@kind, NCHAR(30), @question, NCHAR(30), @opts));
END;
GO

CREATE OR ALTER FUNCTION jev.row_hash (@row nvarchar(max))
RETURNS binary(32)
WITH SCHEMABINDING
AS
BEGIN
    RETURN HASHBYTES('SHA2_256', @row);
END;
GO

-- Full answer JSON, or NULL when the row has not been judged for this question (run jev.judge).
CREATE OR ALTER FUNCTION jev.eval (@row nvarchar(max), @question nvarchar(4000), @kind varchar(10), @options nvarchar(max))
RETURNS nvarchar(max)
AS
BEGIN
    RETURN (SELECT answer FROM jev.answers
            WHERE question_key = jev.question_key(@question, @kind, @options) AND row_hash = jev.row_hash(@row));
END;
GO

-- Probability (0..1) that the row satisfies the condition.
CREATE OR ALTER FUNCTION jev.prob (@row nvarchar(max), @condition nvarchar(4000))
RETURNS float
AS
BEGIN
    RETURN CAST(JSON_VALUE(jev.eval(@row, @condition, 'noul', NULL), '$.noul') AS float);
END;
GO

-- 1 when the row satisfies the condition (threshold: argument, then jev.settings 'threshold', then 0.5).
CREATE OR ALTER FUNCTION jev.matches (@row nvarchar(max), @condition nvarchar(4000), @threshold float = NULL)
RETURNS bit
AS
BEGIN
    DECLARE @p float = jev.prob(@row, @condition);
    IF @p IS NULL RETURN NULL;
    RETURN CASE WHEN @p >= COALESCE(@threshold,
                    (SELECT TRY_CAST(value AS float) FROM jev.settings WHERE name = N'threshold'), 0.5)
                THEN 1 ELSE 0 END;
END;
GO

-- The most likely option (@options: JSON array, e.g. N'["billing","technical","sales"]').
CREATE OR ALTER FUNCTION jev.choice (@row nvarchar(max), @question nvarchar(4000), @options nvarchar(max))
RETURNS nvarchar(4000)
AS
BEGIN
    RETURN JSON_VALUE(jev.eval(@row, @question, 'choice', @options), '$.choice');
END;
GO

-- Probability-weighted level index 0 .. n-1 (@levels: JSON array, lowest first).
CREATE OR ALTER FUNCTION jev.score (@row nvarchar(max), @question nvarchar(4000), @levels nvarchar(max))
RETURNS float
AS
BEGIN
    RETURN CAST(JSON_VALUE(jev.eval(@row, @question, 'score', @levels), '$.score') AS float);
END;
GO

CREATE OR ALTER FUNCTION jev.score_norm (@row nvarchar(max), @question nvarchar(4000), @levels nvarchar(max))
RETURNS float
AS
BEGIN
    DECLARE @n int = (SELECT COUNT(*) FROM OPENJSON(@levels));
    RETURN jev.score(@row, @question, @levels) / CASE WHEN @n > 2 THEN @n - 1 ELSE 1 END;
END;
GO

-- Set-based reads: every judged row of a question, to join on jev.row_hash(<FOR JSON of the row>).
CREATE OR ALTER FUNCTION jev.answers_for (@question nvarchar(4000), @kind varchar(10), @options nvarchar(max))
RETURNS TABLE
AS
RETURN
    SELECT a.row_hash,
           CAST(JSON_VALUE(a.answer, '$.noul') AS float)       AS prob,
           JSON_VALUE(a.answer, '$.choice')                    AS choice,
           CAST(JSON_VALUE(a.answer, '$.score') AS float)      AS score,
           CAST(JSON_VALUE(a.answer, '$.confidence') AS float) AS confidence,
           a.answer
    FROM jev.answers AS a
    WHERE a.question_key = jev.question_key(@question, @kind, @options);
GO

CREATE OR ALTER PROCEDURE jev.configure
    @gateway_url nvarchar(4000) = NULL,
    @credential  nvarchar(4000) = NULL,
    @threshold   float          = NULL
AS
BEGIN
    SET NOCOUNT ON;
    DECLARE @s TABLE (name sysname, value nvarchar(4000));
    INSERT @s VALUES (N'gateway_url', @gateway_url), (N'credential', @credential),
                     (N'threshold', CAST(@threshold AS nvarchar(40)));
    MERGE jev.settings AS t
    USING (SELECT name, value FROM @s WHERE value IS NOT NULL) AS s ON t.name = s.name
    WHEN MATCHED THEN UPDATE SET value = s.value
    WHEN NOT MATCHED THEN INSERT (name, value) VALUES (s.name, s.value);
    SELECT name, CASE WHEN name = N'credential' THEN N'(set)' ELSE value END AS value FROM jev.settings;
END;
GO

-- Judge every row of @source (table or view) that has no answer yet for this question.
--   @where       optional filter in T-SQL, applied before anything is sent (cheap predicates first)
--   @batch_rows  rows per gateway request; the gateway splits them into forward passes
CREATE OR ALTER PROCEDURE jev.judge
    @source     nvarchar(512),
    @question   nvarchar(4000),
    @kind       varchar(10)   = 'noul',
    @options    nvarchar(max) = NULL,
    @where      nvarchar(max) = NULL,
    @batch_rows int           = 500
AS
BEGIN
    SET NOCOUNT ON;
    DECLARE @obj int = OBJECT_ID(@source);
    IF @obj IS NULL
        THROW 50001, N'jev.judge: @source is not a table or view in this database', 1;
    IF @kind NOT IN ('noul', 'choice', 'score')
        THROW 50002, N'jev.judge: @kind must be noul, choice or score', 1;
    IF @kind <> 'noul' AND (ISJSON(@options) = 0 OR (SELECT COUNT(*) FROM OPENJSON(@options)) < 2)
        THROW 50003, N'jev.judge: choice and score need @options as a JSON array of at least two strings', 1;
    DECLARE @url nvarchar(4000) = (SELECT value FROM jev.settings WHERE name = N'gateway_url');
    DECLARE @cred nvarchar(4000) = (SELECT value FROM jev.settings WHERE name = N'credential');
    IF @url IS NULL
        THROW 50004, N'jev.judge: no gateway. EXEC jev.configure @gateway_url = N''https://.../sqlserver''', 1;

    DECLARE @qkey binary(32) = jev.question_key(@question, @kind, @options);
    DECLARE @name nvarchar(600) = QUOTENAME(OBJECT_SCHEMA_NAME(@obj)) + N'.' + QUOTENAME(OBJECT_NAME(@obj));

    CREATE TABLE #rows (row_json nvarchar(max) NOT NULL);
    DECLARE @sql nvarchar(max) = N'INSERT #rows (row_json) SELECT (SELECT t.* FOR JSON PATH, WITHOUT_ARRAY_WRAPPER) FROM '
        + @name + N' AS t' + COALESCE(N' WHERE ' + @where, N'') + N';';
    EXEC sys.sp_executesql @sql;

    CREATE TABLE #todo (n int NOT NULL PRIMARY KEY, row_hash binary(32) NOT NULL, row_json nvarchar(max) NOT NULL);
    INSERT #todo (n, row_hash, row_json)
    SELECT ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) - 1, h, row_json
    FROM (SELECT jev.row_hash(row_json) AS h, row_json,
                 ROW_NUMBER() OVER (PARTITION BY jev.row_hash(row_json) ORDER BY (SELECT NULL)) AS dup
          FROM #rows) AS r
    WHERE dup = 1
      AND NOT EXISTS (SELECT 1 FROM jev.answers AS a WHERE a.question_key = @qkey AND a.row_hash = r.h);

    DECLARE @total int = (SELECT COUNT(*) FROM #rows), @n int = (SELECT COUNT(*) FROM #todo);
    DECLARE @lo int = 0, @payload nvarchar(max), @resp nvarchar(max), @rc int, @status int, @msg nvarchar(2048);
    WHILE @lo < @n
    BEGIN
        SET @payload = (
            SELECT @question AS question, @kind AS kind, JSON_QUERY(@options) AS options,
                   JSON_QUERY(N'[' + (SELECT STRING_AGG(row_json, N',') WITHIN GROUP (ORDER BY n)
                                      FROM #todo WHERE n >= @lo AND n < @lo + @batch_rows) + N']') AS rows
            FOR JSON PATH, WITHOUT_ARRAY_WRAPPER);
        SET @resp = NULL;
        IF @cred IS NULL
            EXEC @rc = sys.sp_invoke_external_rest_endpoint
                 @url = @url, @method = 'POST', @payload = @payload, @timeout = 230, @response = @resp OUTPUT;
        ELSE
            EXEC @rc = sys.sp_invoke_external_rest_endpoint
                 @url = @url, @method = 'POST', @payload = @payload, @credential = @cred, @timeout = 230,
                 @response = @resp OUTPUT;
        SET @status = TRY_CAST(JSON_VALUE(@resp, '$.response.status.http.code') AS int);
        IF @rc <> 0 OR @status <> 200
        BEGIN
            SET @msg = CONCAT(N'jev.judge: gateway returned HTTP ', @status, N': ',
                              COALESCE(JSON_VALUE(@resp, '$.result.error'), LEFT(@resp, 1000)));
            THROW 50005, @msg, 1;
        END;
        INSERT jev.answers (question_key, row_hash, answer)
        SELECT @qkey, t.row_hash, a.value
        FROM OPENJSON(@resp, '$.result.answers') AS a
        JOIN #todo AS t ON t.n = @lo + CAST(a.[key] AS int);
        SET @lo += @batch_rows;
    END;
    SELECT @total AS rows_scanned, @n AS rows_judged, @total - @n AS rows_reused;
END;
GO

-- Forget answers: one question, or everything.
CREATE OR ALTER PROCEDURE jev.forget
    @question nvarchar(4000) = NULL,
    @kind     varchar(10)    = 'noul',
    @options  nvarchar(max)  = NULL
AS
BEGIN
    SET NOCOUNT ON;
    IF @question IS NULL
        DELETE FROM jev.answers;
    ELSE
        DELETE FROM jev.answers WHERE question_key = jev.question_key(@question, @kind, @options);
    SELECT @@ROWCOUNT AS answers_removed;
END;
GO
