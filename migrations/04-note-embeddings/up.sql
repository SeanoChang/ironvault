CREATE TABLE note_embeddings (
    note_id    TEXT PRIMARY KEY,
    embedding  BLOB NOT NULL,
    model      TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
