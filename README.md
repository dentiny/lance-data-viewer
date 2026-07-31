# Lance Data Viewer

A stateless, read-only web UI for inspecting a Lance dataset from a URI.
The backend opens the URI supplied by the browser, so the container does not
need a preconfigured data directory or persistent volume.

## Quick start

Build the image:

```bash
docker build -f docker/Dockerfile \
  -t lance-data-viewer:dev .
```

Start the viewer with one command:

```bash
docker run --rm -p 8080:8080 lance-data-viewer:dev
```

Open <http://localhost:8080>, enter a dataset URI, and select **Connect**.
Examples:

```text
s3://my-bucket/path/events.lance
gs://my-bucket/path/events.lance
az://my-container/path/events.lance
```

The URI is request-scoped. It is not stored as global backend state, so one
deployment can serve multiple users and replicas.

## Cloud access

Lance resolves remote object-store URIs directly. Give the container or
Kubernetes workload access using the cloud provider's workload identity:

- AWS IAM role / IRSA for `s3://`
- Google Workload Identity for `gs://`
- Azure managed identity for `az://`

The current UI accepts only a dataset URI. It does not accept, store, or
forward cloud credentials.

## Kubernetes

Publish the image to a registry and deploy it as a stateless service:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lance-data-viewer
spec:
  replicas: 2
  selector:
    matchLabels:
      app: lance-data-viewer
  template:
    metadata:
      labels:
        app: lance-data-viewer
    spec:
      containers:
        - name: viewer
          image: ghcr.io/dentiny/lance-data-viewer:latest
          ports:
            - name: http
              containerPort: 8080
          readinessProbe:
            httpGet:
              path: /healthz
              port: http
          livenessProbe:
            httpGet:
              path: /healthz
              port: http
---
apiVersion: v1
kind: Service
metadata:
  name: lance-data-viewer
spec:
  selector:
    app: lance-data-viewer
  ports:
    - port: 80
      targetPort: http
```

No volume is required for remote datasets. Configure the pod's service
account or workload identity separately when the dataset is private.

## Optional local dataset access

A dataset on the backend host must be visible inside the container. Mount only
the directory needed for local development:

```bash
docker run --rm -p 8080:8080 \
  -v "$HOME/Desktop/example_lance:/datasets:ro" \
  lance-data-viewer:dev
```

Then enter this URI in the UI:

```text
/datasets/multimedia.lance
```

This is optional local development behavior, not a startup requirement.

## API

Every dataset endpoint requires a `uri` query parameter:

```bash
curl --get http://localhost:8080/dataset \
  --data-urlencode 'uri=s3://my-bucket/path/events.lance'

curl --get http://localhost:8080/dataset/schema \
  --data-urlencode 'uri=s3://my-bucket/path/events.lance'

curl --get http://localhost:8080/dataset/rows \
  --data-urlencode 'uri=s3://my-bucket/path/events.lance' \
  --data-urlencode 'limit=50' \
  --data-urlencode 'offset=0'

curl --get http://localhost:8080/dataset/cell \
  --data-urlencode 'uri=s3://my-bucket/path/events.lance' \
  --data-urlencode 'column=media' \
  --data-urlencode 'index=0'

curl --get http://localhost:8080/dataset/sql \
  --data-urlencode 'uri=s3://my-bucket/path/events.lance' \
  --data-urlencode 'query=SELECT * FROM dataset LIMIT 20'
```

Available endpoints:

- `GET /healthz`
- `GET /dataset`
- `GET /dataset/schema`
- `GET /dataset/columns`
- `GET /dataset/rows`
- `GET /dataset/cell`
- `GET /dataset/sql`
- `GET /dataset/vector/preview`

## Features

- Direct Lance dataset access through local or object-store URIs
- Request-scoped connections suitable for concurrent users and replicas
- Schema and column inspection
- Server-side pagination and column filtering
- Native `LanceDataset.sql()` queries against the `dataset` table
- Fixed-size and variable-length vector visualization
- CLIP-512 detection, statistics, sparklines, and tooltips
- Recursive rendering for nested structs and lists
- Read-only operation

Binary values are detected from their file signatures. Recognized images,
audio, and videos are rendered with native browser controls; unknown binary
values retain the existing UTF-8 or base64 fallback. Common formats include
PNG, JPEG, GIF, WebP, WAV, MP3, FLAC, Ogg, MP4, WebM, AVI, and MPEG.
Blob-backed cells are loaded lazily as they approach the viewport.

## Development

Run the backend tests with Python 3.11:

```bash
pip install -r backend/requirements.txt
pip install pytest httpx2
cd backend
python -m pytest tests/ -v
```

The Docker image serves the static frontend and FastAPI backend on port 8080.
Set `PORT` to change the port inside the container.

## Security

- The service is read-only.
- Dataset URIs are supplied by users and opened by the backend.
- Do not expose the service publicly without authentication and URI access
  controls.
- Restrict the workload identity to the buckets and prefixes users are
  allowed to inspect.

Credential input, URI allowlists, and multi-tenant authorization are planned
separately.

## License

MIT
