#!/usr/bin/env python3

import os
import logging
from pathlib import Path
from typing import Optional

import lance
import pyarrow as pa
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from serialize_value import serialize_value

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _read_app_version() -> str:
    here = Path(__file__).resolve().parent
    for candidate in (here / "VERSION", here.parent / "VERSION"):
        if candidate.exists():
            return candidate.read_text().strip()
    return "0.0.0-dev"


APP_VERSION = _read_app_version()

app = FastAPI(
    title="Lance Data Viewer",
    description="Read-only web viewer for Lance datasets",
    version=APP_VERSION,
)

@app.on_event("startup")
async def startup_event():
    """Log version information on startup"""
    logger.info(f"Lance Data Viewer v{APP_VERSION}")
    logger.info(f"Lance: {lance.__version__}, PyArrow: {pa.__version__}")
    logger.info("Waiting for a dataset URI from the UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

MAX_LIMIT = 1000
BLOB_DESCRIPTOR_FIELDS = {"position", "size"}
BLOB_V2_DESCRIPTOR_FIELDS = {
    "kind",
    "position",
    "size",
    "blob_id",
    "blob_uri",
}

def open_dataset(uri: str):
    """Open a Lance dataset from any URI supported by Lance."""
    dataset_uri = uri.strip()
    if not dataset_uri:
        raise HTTPException(status_code=400, detail="Dataset URI is required")

    try:
        return lance.dataset(dataset_uri)
    except Exception as error:
        logger.warning("Failed to open dataset URI: %s", error)
        raise HTTPException(status_code=400, detail="Unable to open dataset URI")


def describe_schema(schema):
    """Build the schema and column metadata used by the viewer."""
    fields = []
    columns = []
    for field in schema:
        is_vector = (
            (pa.types.is_list(field.type) or pa.types.is_fixed_size_list(field.type))
            and pa.types.is_floating(field.type.value_type)
        )
        field_info = {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        if is_vector:
            field_info["vector_dim"] = None
        fields.append(field_info)

        column = {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
            "is_vector": is_vector,
        }
        if is_vector:
            column["dim"] = None
        columns.append(column)

    metadata = {
        key.decode("utf-8", errors="replace"): value.decode("utf-8", errors="replace")
        for key, value in (schema.metadata or {}).items()
    }
    return {"fields": fields, "metadata": metadata, "columns": columns}


def _serialize_blob_reference(value, blob_context, path):
    if blob_context is None or not pa.types.is_struct(value.type):
        return False, None

    field_names = {field.name for field in value.type}
    if field_names not in (BLOB_DESCRIPTOR_FIELDS, BLOB_V2_DESCRIPTOR_FIELDS):
        return False, None

    descriptor = value.as_py()
    if field_names == BLOB_DESCRIPTOR_FIELDS:
        is_null = descriptor["position"] == 1 and descriptor["size"] == 0
    else:
        is_null = (
            descriptor["kind"] == 0
            and descriptor["position"] == 0
            and descriptor["size"] == 0
            and descriptor["blob_id"] == 0
            and not descriptor["blob_uri"]
        )
    if is_null:
        return True, None

    return True, {
        "type": "blob_ref",
        "column": blob_context["column"],
        "index": blob_context["index"],
        "path": list(path),
        "size": descriptor["size"],
    }


def serialize_arrow_value(value, blob_context=None, path=()):
    try:
        # Stop immediately if the Arrow scalar is null
        if value is None or not getattr(value, "is_valid", True):
            return None

        # 1. Handle Vector columns (Top-level OR nested)
        if (pa.types.is_list(value.type) or pa.types.is_fixed_size_list(value.type)) and getattr(value.type, "value_type", None) and pa.types.is_floating(value.type.value_type):
            try:
                vec = value.as_py()
                if vec is None:
                    return None

                if not isinstance(vec, (list, tuple)) or len(vec) == 0:
                    return {"type": "vector", "error": "Invalid vector data"}

                valid_values = []
                for v in vec:
                    if v is not None and isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf'))):
                        valid_values.append(float(v))
                    else:
                        valid_values.append(0.0)

                if not valid_values:
                    return {"type": "vector", "error": "No valid numeric values in vector"}

                norm = float(sum(x*x for x in valid_values) ** 0.5) if valid_values else 0.0
                vec_min = float(min(valid_values)) if valid_values else 0.0
                vec_max = float(max(valid_values)) if valid_values else 0.0
                vec_mean = float(sum(valid_values) / len(valid_values)) if valid_values else 0.0

                is_clip_vector = len(valid_values) == 512

                result = {
                    "type": "vector",
                    "dim": len(valid_values),
                    "norm": norm,
                    "min": vec_min,
                    "max": vec_max,
                    "mean": vec_mean,
                    "preview": valid_values[:32],
                }

                if is_clip_vector:
                    result["model"] = "likely_clip"
                    result["description"] = "512-dimensional CLIP embedding"
                    result["stats"] = {
                        "normalized": abs(norm - 1.0) < 0.01,
                        "sparsity": sum(1 for x in valid_values if abs(x) < 0.01) / len(valid_values),
                        "positive_ratio": sum(1 for x in valid_values if x > 0) / len(valid_values)
                    }
                return result
            except Exception as vec_error:
                logger.warning(f"Error processing vector data: {vec_error}")
                return {"type": "vector", "error": f"Vector processing failed: {str(vec_error)}"}

        # 2. Keep blob payloads lazy until their cells enter the viewport.
        is_blob, blob_reference = _serialize_blob_reference(
            value, blob_context, path
        )
        if is_blob:
            return blob_reference

        # 3. Handle Structs recursively to catch vectors hidden inside objects
        if pa.types.is_struct(value.type):
            result = {}
            for field in value.type:
                # In PyArrow, value[field.name] fetches the nested pa.Scalar
                result[field.name] = serialize_arrow_value(
                    value[field.name],
                    blob_context,
                    (*path, field.name),
                )
            return result

        # 4. Handle Lists recursively (e.g., Arrays of Structs containing Vectors)
        if pa.types.is_list(value.type) or pa.types.is_large_list(value.type) or pa.types.is_fixed_size_list(value.type):
            result = []
            for index, item in enumerate(value):
                result.append(
                    serialize_arrow_value(
                        item,
                        blob_context,
                        (*path, index),
                    )
                )
            return result

        # 5. Fallback to normal serialization for strings, ints, dates, etc.
        return serialize_value(value)
    except Exception as e:
        logger.warning(f"Error serializing value: {e}")
        return {"error": f"Serialization failed: {str(e)}"}


def serialize_arrow_table(table, row_offset=0, lazy_blobs=False):
    rows = []
    for row_index in range(table.num_rows):
        row = {}
        for column_index, column_name in enumerate(table.column_names):
            try:
                value = table.column(column_index)[row_index]
                blob_context = None
                if lazy_blobs:
                    blob_context = {
                        "column": column_name,
                        "index": row_offset + row_index,
                    }
                row[column_name] = serialize_arrow_value(
                    value,
                    blob_context=blob_context,
                )
            except Exception as serialize_error:
                logger.warning(
                    "Failed to serialize column %s at row %s: %s",
                    column_name,
                    row_index,
                    serialize_error,
                )
                row[column_name] = {"error": "Failed to read value"}
        rows.append(row)
    return rows


@app.get("/healthz")
async def health_check():
    try:
        lance_version = lance.__version__
        pyarrow_version = pa.__version__

        compat = {
            "vector_preview": True,
            "remote_dataset_uri": True,
            "sql": True,
        }

        build_tag = f"app-{APP_VERSION}_lance-{lance_version}"

        return {
            "ok": True,
            "app_version": APP_VERSION,
            "lance_version": lance_version,
            "pyarrow_version": pyarrow_version,
            "build_tag": build_tag,
            "compat": compat
        }
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {"ok": False, "error": str(e)}


# Lance exposes synchronous I/O, so FastAPI runs this handler in its thread pool.
@app.get("/dataset")
def get_dataset_info(uri: str = Query(min_length=1)):
    dataset = open_dataset(uri)
    try:
        description = describe_schema(dataset.schema)
        return {
            "uri": uri,
            "version": dataset.version,
            **description,
        }
    except Exception as error:
        logger.warning("Failed to inspect dataset: %s", error)
        raise HTTPException(status_code=400, detail="Unable to inspect dataset")


@app.get("/dataset/schema")
async def get_dataset_schema(uri: str = Query(min_length=1)):
    description = describe_schema(open_dataset(uri).schema)
    return {
        "fields": description["fields"],
        "metadata": description["metadata"],
    }


@app.get("/dataset/columns")
async def get_dataset_columns(uri: str = Query(min_length=1)):
    description = describe_schema(open_dataset(uri).schema)
    return {"columns": description["columns"]}


# Keep row I/O off the event loop so it can overlap metadata loading.
@app.get("/dataset/rows")
def get_dataset_rows(
    uri: str = Query(min_length=1),
    limit: int = Query(default=50, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    columns: Optional[str] = Query(default=None),
    lazy_blobs: bool = Query(default=False),
):
    dataset = open_dataset(uri)
    schema = dataset.schema

    column_list = None
    if columns:
        column_list = [column.strip() for column in columns.split(",") if column.strip()]
        schema_columns = set(schema.names)
        invalid_columns = [column for column in column_list if column not in schema_columns]
        if invalid_columns:
            raise HTTPException(status_code=400, detail=f"Invalid columns: {invalid_columns}")

    try:
        total_count = dataset.count_rows()
        result_table = dataset.scanner(
            columns=column_list,
            offset=offset,
            limit=limit,
            blob_handling=(
                "blobs_descriptions" if lazy_blobs else "all_binary"
            ),
        ).to_table()
        logger.info(
            "Read %s rows (offset=%s, limit=%s) from dataset",
            result_table.num_rows,
            offset,
            limit,
        )
    except Exception as read_error:
        logger.warning("Failed to read dataset rows: %s", read_error)
        result_table = pa.table(
            {
                "error": ["Unable to read dataset"],
                "dataset": [uri],
                "details": [f"Error: {str(read_error)[:200]}"],
            }
        )
        total_count = 1

    return {
        "rows": serialize_arrow_table(
            result_table,
            row_offset=offset,
            lazy_blobs=lazy_blobs,
        ),
        "total": total_count,
        "limit": limit,
        "offset": offset,
    }


@app.get("/dataset/cell")
def get_dataset_cell(
    uri: str = Query(min_length=1),
    column: str = Query(min_length=1),
    index: int = Query(ge=0),
):
    dataset = open_dataset(uri)
    if column not in dataset.schema.names:
        raise HTTPException(status_code=400, detail=f"Invalid column: {column}")

    try:
        table = dataset.scanner(
            columns=[column],
            offset=index,
            limit=1,
            blob_handling="all_binary",
        ).to_table()
    except Exception as error:
        logger.warning("Failed to read dataset cell: %s", error)
        raise HTTPException(status_code=400, detail="Unable to read dataset cell")

    if table.num_rows == 0:
        raise HTTPException(status_code=404, detail="Dataset row not found")

    return {"value": serialize_arrow_value(table.column(0)[0])}


@app.get("/dataset/sql")
async def query_dataset_sql(
    uri: str = Query(min_length=1),
    query: str = Query(min_length=1),
    limit: int = Query(default=500, ge=1, le=MAX_LIMIT),
):
    statement = query.strip().rstrip(";").strip()
    if not statement.lower().startswith(("select", "with")):
        raise HTTPException(
            status_code=400,
            detail="Only SELECT or WITH queries are supported",
        )

    dataset = open_dataset(uri)
    limited_query = (
        f"SELECT * FROM ({statement}) AS viewer_query LIMIT {limit + 1}"
    )
    try:
        result = (
            dataset.sql(limited_query)
            .build()
            .to_stream_reader()
            .read_all()
        )
    except Exception as error:
        logger.warning("SQL query failed: %s", error)
        raise HTTPException(
            status_code=400,
            detail=f"SQL query failed: {str(error)[:500]}",
        )

    truncated = result.num_rows > limit
    if truncated:
        result = result.slice(0, limit)
    return {
        "rows": serialize_arrow_table(result),
        "columns": result.column_names,
        "count": result.num_rows,
        "truncated": truncated,
        "limit": limit,
    }


@app.get("/dataset/vector/preview")
async def get_vector_preview(
    uri: str = Query(min_length=1),
    column: str = Query(min_length=1),
    limit: int = Query(default=100, ge=1, le=MAX_LIMIT),
):
    dataset = open_dataset(uri)
    schema = dataset.schema
    if column not in schema.names:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found")

    field = schema.field(column)
    if not (
        (pa.types.is_list(field.type) or pa.types.is_fixed_size_list(field.type))
        and pa.types.is_floating(field.type.value_type)
    ):
        raise HTTPException(
            status_code=400, detail=f"Column '{column}' is not a vector column"
        )

    row_count = min(limit, dataset.count_rows())
    result = dataset.take(list(range(row_count)), columns=[column])
    vectors = result.column(0).to_pylist()
    valid_vectors = [vector for vector in vectors if vector is not None]
    if not valid_vectors:
        return {"stats": None, "preview": []}

    all_values = [value for vector in valid_vectors for value in vector]
    stats = {
        "count": len(valid_vectors),
        "dim": len(valid_vectors[0]),
        "min": min(all_values) if all_values else 0,
        "max": max(all_values) if all_values else 0,
        "mean": sum(all_values) / len(all_values) if all_values else 0,
    }
    preview = [
        {
            "norm": float(sum(value * value for value in vector) ** 0.5),
            "sample": vector[:32],
        }
        for vector in valid_vectors[:20]
        if vector
    ]
    return {"stats": stats, "preview": preview}

# Mount static files - use vanilla version by default
# In production, Docker copies vanilla files to /web
# For local development, serve from web/vanilla
static_dir = "/web"
if not os.path.exists(static_dir):
    # Local development - serve vanilla version
    static_dir = os.path.join(os.path.dirname(__file__), "..", "web", "vanilla")

if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    # Log version information on startup
    logger.info(f"Lance Data Viewer v{APP_VERSION}")
    logger.info(f"Lance: {lance.__version__}, PyArrow: {pa.__version__}")

    uvicorn.run(app, host="0.0.0.0", port=8080)