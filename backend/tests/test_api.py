"""API tests for request-scoped Lance dataset URIs."""

import base64
import inspect
from types import SimpleNamespace

import lance
import pyarrow as pa
import pytest

from conftest import CLIP_DIM, ROWS, VEC_DIM


def api_get(client, path, uri, **params):
    return client.get(path, params={"uri": str(uri), **params})


def test_healthz_reports_versions(client):
    body = client.get("/healthz").json()
    assert body["ok"] is True
    assert body["lance_version"] == lance.__version__
    assert body["build_tag"] == f"app-{body['app_version']}_lance-{lance.__version__}"


def test_healthz_compat_flags(client):
    compat = client.get("/healthz").json()["compat"]
    assert compat["vector_preview"] is True
    assert compat["remote_dataset_uri"] is True


def test_dataset_info(client, sample_uri):
    response = api_get(client, "/dataset", sample_uri)
    assert response.status_code == 200
    body = response.json()
    assert body["uri"] == sample_uri
    assert "rows" not in body
    assert [field["name"] for field in body["fields"]] == [
        "id",
        "text",
        "score",
        "blob",
        "vec",
        "embedding",
    ]
    assert [column["name"] for column in body["columns"]] == [
        "id",
        "text",
        "score",
        "blob",
        "vec",
        "embedding",
    ]


def test_dataset_uri_is_required(client):
    assert client.get("/dataset").status_code == 422
    assert client.get("/dataset/metadata").status_code == 422
    assert client.get("/dataset/schema").status_code == 422
    assert client.get("/dataset/rows").status_code == 422


def test_invalid_dataset_uri_returns_400(client):
    response = api_get(client, "/dataset", "s3://bucket/does-not-exist.lance")
    assert response.status_code == 400
    assert response.json()["detail"] == "Unable to open dataset URI"


def test_uri_is_passed_to_lance_unchanged(client, monkeypatch):
    import app as app_module

    received = []

    class FakeDataset:
        version = 7
        schema = pa.schema([("id", pa.int64())])

    monkeypatch.setattr(
        app_module.lance,
        "dataset",
        lambda uri: received.append(uri) or FakeDataset(),
    )
    uri = "s3://example-bucket/path/data.lance"
    response = api_get(client, "/dataset", uri)
    assert response.status_code == 200
    assert received == [uri]


def test_schema_fields(client, sample_uri):
    body = api_get(client, "/dataset/schema", sample_uri).json()
    assert "metadata" in body
    fields = {field["name"]: field for field in body["fields"]}
    assert set(fields) == {"id", "text", "score", "blob", "vec", "embedding"}
    assert fields["id"]["type"] == "int64"
    assert fields["id"]["nullable"] is True
    assert fields["vec"]["type"] == "list<item: float>"
    assert fields["embedding"]["type"].startswith("fixed_size_list")
    assert str(CLIP_DIM) in fields["embedding"]["type"]
    assert "vector_dim" in fields["vec"]
    assert "vector_dim" not in fields["id"]


def test_metadata_combines_schema_and_columns(client, sample_uri):
    body = api_get(client, "/dataset/metadata", sample_uri).json()
    assert {field["name"] for field in body["fields"]} == {
        "id", "text", "score", "blob", "vec", "embedding"
    }
    columns = {column["name"]: column for column in body["columns"]}
    assert columns["vec"]["is_vector"] is True
    assert columns["id"]["is_vector"] is False


def test_metadata_serializes_utf8_and_binary_schema_metadata(client, monkeypatch):
    import app as app_module

    schema = pa.schema(
        [pa.field("id", pa.int64())],
        metadata={
            "café".encode(): "naïve".encode(),
            b"binary": b"\xff\xfe\x01\x02",
        },
    )
    monkeypatch.setattr(
        app_module,
        "open_dataset",
        lambda _uri: SimpleNamespace(schema=schema),
    )

    metadata = api_get(client, "/dataset/metadata", "test.lance").json()["metadata"]
    assert metadata["café"] == "naïve"
    assert metadata["binary"] == base64.b64encode(b"\xff\xfe\x01\x02").decode()


def test_dataset_io_handlers_are_synchronous():
    import app as app_module

    handlers = (
        app_module.get_dataset_info,
        app_module.get_dataset_metadata,
        app_module.get_dataset_schema,
        app_module.get_dataset_columns,
        app_module.get_dataset_rows,
        app_module.get_vector_preview,
    )
    assert all(not inspect.iscoroutinefunction(handler) for handler in handlers)


def test_schema_readable_on_corrupted_dataset(client, broken_uri):
    response = api_get(client, "/dataset/schema", broken_uri)
    assert response.status_code == 200
    assert [field["name"] for field in response.json()["fields"]] == ["id"]


def test_columns_vector_flags(client, sample_uri):
    response = api_get(client, "/dataset/columns", sample_uri)
    assert response.status_code == 200
    columns = {column["name"]: column for column in response.json()["columns"]}
    assert columns["vec"]["is_vector"] is True
    assert columns["vec"]["dim"] is None
    assert columns["embedding"]["is_vector"] is True
    assert columns["id"]["is_vector"] is False
    assert "dim" not in columns["id"]


def test_rows_defaults(client, sample_uri):
    response = api_get(client, "/dataset/rows", sample_uri)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == ROWS
    assert body["limit"] == 50
    assert body["offset"] == 0
    assert len(body["rows"]) == ROWS


def test_rows_pagination(client, sample_uri):
    body = api_get(
        client, "/dataset/rows", sample_uri, limit=3, offset=0
    ).json()
    assert [row["id"] for row in body["rows"]] == [0, 1, 2]

    body = api_get(
        client, "/dataset/rows", sample_uri, limit=3, offset=8
    ).json()
    assert [row["id"] for row in body["rows"]] == [8, 9]


def test_rows_offset_past_end(client, sample_uri):
    body = api_get(client, "/dataset/rows", sample_uri, offset=ROWS).json()
    assert body["rows"] == []
    assert body["total"] == ROWS


def test_rows_limit_bounds(client, sample_uri):
    assert api_get(client, "/dataset/rows", sample_uri, limit=0).status_code == 422
    assert api_get(client, "/dataset/rows", sample_uri, limit=1000).status_code == 200
    assert api_get(client, "/dataset/rows", sample_uri, limit=1001).status_code == 422
    assert api_get(client, "/dataset/rows", sample_uri, offset=-1).status_code == 422


def test_rows_column_filtering(client, sample_uri):
    body = api_get(
        client, "/dataset/rows", sample_uri, columns="id,text"
    ).json()
    assert all(set(row) == {"id", "text"} for row in body["rows"])


def test_rows_invalid_column_returns_400(client, sample_uri):
    response = api_get(
        client, "/dataset/rows", sample_uri, columns="id,nope"
    )
    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


def test_rows_scalar_and_binary_serialization(client, sample_uri):
    rows = api_get(client, "/dataset/rows", sample_uri).json()["rows"]
    assert rows[0]["id"] == 0
    assert rows[0]["text"] == "row 0"
    assert rows[1]["score"] == 1.5
    assert rows[3]["text"] is None
    assert rows[0]["blob"] == "hello"
    assert rows[1]["blob"] == base64.b64encode(b"\xff\xfe\x01\x02").decode()


def test_rows_return_lazy_references_for_blob_lists(client, media_list_uri):
    rows = api_get(client, "/dataset/rows", media_list_uri).json()["rows"]

    media = rows[0]["media"]
    assert len(media) == 3
    assert all(item["type"] == "blob_ref" for item in media)
    assert all(item["column"] == "media" for item in media)
    assert all(item["index"] == 0 for item in media)
    assert [item["path"] for item in media] == [[0], [1], [2]]
    assert all("base64" not in item for item in media)


def test_cell_materializes_media_in_blob_lists(client, media_list_uri):
    response = api_get(
        client,
        "/dataset/cell",
        media_list_uri,
        column="media",
        index=0,
    )
    assert response.status_code == 200
    media = response.json()["value"]
    assert media[0]["type"] == "media"
    assert media[0]["media_type"] == "image"
    assert media[0]["mime_type"] == "image/jpeg"
    assert media[1]["media_type"] == "audio"
    assert media[1]["mime_type"] == "audio/wav"
    assert media[2]["media_type"] == "video"
    assert media[2]["mime_type"] == "video/mp4"


def test_cell_validates_column_and_row(client, sample_uri):
    invalid_column = api_get(
        client,
        "/dataset/cell",
        sample_uri,
        column="missing",
        index=0,
    )
    assert invalid_column.status_code == 400

    missing_row = api_get(
        client,
        "/dataset/cell",
        sample_uri,
        column="id",
        index=ROWS,
    )
    assert missing_row.status_code == 404


def test_rows_vector_serialization(client, sample_uri):
    vector = api_get(client, "/dataset/rows", sample_uri).json()["rows"][0]["vec"]
    assert vector["type"] == "vector"
    assert vector["dim"] == VEC_DIM
    assert vector["preview"] == [0.0, -1.0, 0.5, 2.0]
    assert vector["norm"] == pytest.approx(5.25 ** 0.5)
    assert vector["mean"] == pytest.approx(0.375)


def test_rows_null_vector(client, sample_uri, vec_nulls_preserved):
    vector = api_get(client, "/dataset/rows", sample_uri).json()["rows"][5]["vec"]
    if vec_nulls_preserved:
        assert vector is None
    else:
        assert vector == {"type": "vector", "error": "Invalid vector data"}


def test_rows_clip_detection(client, sample_uri):
    embedding = api_get(
        client, "/dataset/rows", sample_uri
    ).json()["rows"][0]["embedding"]
    assert embedding["type"] == "vector"
    assert embedding["dim"] == CLIP_DIM
    assert embedding["model"] == "likely_clip"
    assert embedding["norm"] == pytest.approx(1.0, abs=1e-3)


def test_rows_graceful_degradation(client, broken_uri):
    body = api_get(client, "/dataset/rows", broken_uri).json()
    assert body["total"] == 1
    assert body["rows"][0]["error"] == "Unable to read dataset"
    assert body["rows"][0]["dataset"] == broken_uri


def test_vector_preview_stats(client, sample_uri, vec_nulls_preserved):
    response = api_get(
        client, "/dataset/vector/preview", sample_uri, column="vec"
    )
    assert response.status_code == 200
    body = response.json()
    assert body["stats"]["count"] == (ROWS - 1 if vec_nulls_preserved else ROWS)
    assert body["stats"]["dim"] == VEC_DIM
    assert body["stats"]["min"] == -1.0
    assert body["stats"]["max"] == 9.0
    assert len(body["preview"]) == ROWS - 1


def test_vector_preview_validation(client, sample_uri):
    assert api_get(
        client, "/dataset/vector/preview", sample_uri, column="score"
    ).status_code == 400
    assert api_get(
        client, "/dataset/vector/preview", sample_uri, column="nope"
    ).status_code == 400
    assert api_get(
        client, "/dataset/vector/preview", sample_uri
    ).status_code == 422


def test_native_lance_sql(client, sample_uri):
    response = api_get(
        client,
        "/dataset/sql",
        sample_uri,
        query="SELECT id, text FROM dataset WHERE id >= 8 ORDER BY id",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["columns"] == ["id", "text"]
    assert [row["id"] for row in body["rows"]] == [8, 9]
    assert body["truncated"] is False


def test_native_lance_sql_applies_result_cap(client, sample_uri):
    body = api_get(
        client,
        "/dataset/sql",
        sample_uri,
        query="SELECT id FROM dataset ORDER BY id",
        limit=3,
    ).json()
    assert [row["id"] for row in body["rows"]] == [0, 1, 2]
    assert body["truncated"] is True
    assert body["limit"] == 3


def test_native_lance_sql_is_read_only(client, sample_uri):
    response = api_get(
        client,
        "/dataset/sql",
        sample_uri,
        query="DELETE FROM dataset",
    )
    assert response.status_code == 400
    assert "Only SELECT" in response.json()["detail"]


def test_native_lance_sql_reports_query_errors(client, sample_uri):
    response = api_get(
        client,
        "/dataset/sql",
        sample_uri,
        query="SELECT missing_column FROM dataset",
    )
    assert response.status_code == 400
    assert "SQL query failed" in response.json()["detail"]
