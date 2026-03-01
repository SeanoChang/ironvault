use anyhow::Result;
use chrono::Utc;
use rusqlite::Connection;

/// Bump access tracking for a note.
/// Called after content is returned in read and about commands.
pub fn bump_access(conn: &Connection, note_id: &str) -> Result<()> {
    conn.execute(
        "UPDATE current_notes \
         SET access_count = access_count + 1, \
             last_accessed = ?1 \
         WHERE note_id = ?2",
        rusqlite::params![Utc::now().to_rfc3339(), note_id],
    )?;
    Ok(())
}
