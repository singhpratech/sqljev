# Gateway on Cloud Run (for BigQuery remote functions)

```bash
gcloud builds submit --tag us-docker.pkg.dev/PROJECT/sqljev/sqljev:0.1.0 -f deploy/Dockerfile .
gcloud run deploy sqljev --image us-docker.pkg.dev/PROJECT/sqljev/sqljev:0.1.0 \
  --region us-central1 --no-allow-unauthenticated --cpu 8 --memory 16Gi --concurrency 4 --timeout 300
# GPU (much faster): add --gpu 1 --gpu-type nvidia-l4 --no-cpu-throttling and build with the cu124 TORCH_INDEX.
```

Then grant the BigQuery connection's service account `roles/run.invoker` on the service and run
`sql/bigquery/install.sql` with the service URL.
