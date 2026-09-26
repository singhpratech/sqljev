"""AWS Lambda handler for Redshift Lambda UDFs (CREATE EXTERNAL FUNCTION ... LAMBDA 'sql-jev-laya').

Handler: sqljev.aws_lambda.handler. Configure with SQLJEV_* environment variables on the function, e.g.
SQLJEV_BACKEND=gateway + SQLJEV_API_URL=https://your-gateway/v1/eval, or SQLJEV_BACKEND=jev + TYPESAFE_API_KEY.
The engine (and its answer cache) lives as long as the Lambda container does.
"""
import json

from .core import default_engine
from .gateway import handle_redshift


def handler(event, context=None):
    if isinstance(event, (str, bytes)):
        event = json.loads(event)
    return json.dumps(handle_redshift(default_engine(), event))
