# Behavioral specification

This document defines the observable API and dataset-loading behavior of the
vanilla web viewer.

## Dataset selection

When a user selects a dataset, the frontend:

1. clears the previous column selection;
2. requests `GET /datasets/{dataset_name}/metadata` and
   `GET /datasets/{dataset_name}/rows` concurrently;
3. renders the schema and column controls from the metadata response; and
4. renders the first page from the rows response.

Metadata and row failures are reported independently. The frontend does not
make separate schema and columns requests during normal dataset loading.

The LanceDB APIs used by the dataset handlers are synchronous. Those FastAPI
handlers must therefore be regular `def` functions so FastAPI executes them in
its thread pool instead of blocking the event loop.

## Combined metadata endpoint

`GET /datasets/{dataset_name}/metadata` opens the dataset once and returns:

```json
{
  "fields": [
    {"name": "id", "type": "int64", "nullable": true}
  ],
  "metadata": {
    "description": "Café vectors",
    "binary": "//4BAg=="
  },
  "columns": [
    {
      "name": "id",
      "type": "int64",
      "nullable": true,
      "is_vector": false
    }
  ]
}
```

- `fields` follows the existing `/schema` field representation. Vector fields
  additionally contain `"vector_dim": null`.
- `columns` follows the existing `/columns` representation. Vector columns
  additionally contain `"dim": null`.
- Arrow schema metadata keys and values are bytes. UTF-8 byte sequences are
  decoded as JSON strings, including non-ASCII text. Values that are not valid
  UTF-8 are returned as base64 strings.
- Invalid dataset names return 400. Datasets that cannot be opened return 500.

`GET /datasets/{dataset_name}/schema` and
`GET /datasets/{dataset_name}/columns` remain available for API compatibility
and use the same metadata description and serialization rules.
