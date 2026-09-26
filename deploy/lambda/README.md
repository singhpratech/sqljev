# Lambda for Redshift

The Lambda forwards Redshift's row batches to a gateway (Laya on your GPU/CPU host):

```bash
pip install --target build/ . && (cd build && zip -r ../sqljev-lambda.zip .)
aws lambda create-function --function-name sqljev --runtime python3.12 \
  --handler sqljev.aws_lambda.handler --zip-file fileb://sqljev-lambda.zip --timeout 300 --memory-size 512 \
  --role arn:aws:iam::123456789012:role/sqljev-lambda \
  --environment "Variables={SQLJEV_BACKEND=gateway,SQLJEV_API_URL=https://gateway.example.com/v1/eval,SQLJEV_GATEWAY_TOKEN=...}"
```

The package is stdlib-only, so the zip is tiny. Then run `sql/redshift/install.sql`.
