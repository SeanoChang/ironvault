use anyhow::Result;
use chrono::Utc;
use rusqlite::Connection;

/// Insert or update an embedding for a note.
pub fn upsert_embedding(conn: &Connection, note_id: &str, embedding: &[f32], model: &str) -> Result<()> {
    let blob = embedding_to_blob(embedding);
    conn.execute(
        "INSERT INTO note_embeddings (note_id, embedding, model, updated_at)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(note_id) DO UPDATE SET
             embedding = excluded.embedding,
             model = excluded.model,
             updated_at = excluded.updated_at",
        rusqlite::params![note_id, blob, model, Utc::now().to_rfc3339()],
    )?;
    Ok(())
}

/// Get embedding for a single note. Returns None if not embedded.
pub fn get_embedding(conn: &Connection, note_id: &str) -> Result<Option<Vec<f32>>> {
    let mut stmt = conn.prepare(
        "SELECT embedding FROM note_embeddings WHERE note_id = ?1"
    )?;

    let result = stmt.query_row([note_id], |row| {
        let blob: Vec<u8> = row.get(0)?;
        Ok(blob)
    });

    match result {
        Ok(blob) => Ok(Some(blob_to_embedding(&blob))),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Get all embeddings as (note_id, embedding) pairs.
pub fn get_all_embeddings(conn: &Connection) -> Result<Vec<(String, Vec<f32>)>> {
    let mut stmt = conn.prepare(
        "SELECT ne.note_id, ne.embedding
         FROM note_embeddings ne
         JOIN current_notes cn ON ne.note_id = cn.note_id
         WHERE cn.namespace = 'ark' AND cn.status != 'retracted'"
    )?;

    let rows: Vec<(String, Vec<f32>)> = stmt
        .query_map([], |row| {
            let note_id: String = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            Ok((note_id, blob_to_embedding(&blob)))
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(rows)
}

/// Get note IDs that don't have embeddings yet.
pub fn get_notes_without_embeddings(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        "SELECT cn.note_id FROM current_notes cn
         WHERE cn.namespace = 'ark'
           AND cn.status != 'retracted'
           AND cn.note_id NOT IN (SELECT note_id FROM note_embeddings)"
    )?;

    let rows: Vec<String> = stmt
        .query_map([], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    Ok(rows)
}

/// Check if any embeddings exist in the database.
pub fn has_embeddings(conn: &Connection) -> bool {
    conn.query_row("SELECT COUNT(*) FROM note_embeddings", [], |r| r.get::<_, i64>(0))
        .unwrap_or(0) > 0
}

fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    embedding.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
