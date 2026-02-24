use anyhow::Result;
use rusqlite::Connection;

pub struct NoteMeta {
    pub note_id: String,
    pub title: String,
    pub domain: String,
    pub intent: String,
    pub kind: String,
    pub trust: String,
    pub status: String,
    pub tags: Vec<String>,
    pub updated_at: String,
}

pub struct NoteRef {
    pub fm_hash: String,
    pub md_hash: String,
}

pub fn get_meta(conn: &Connection, note_id: &str) -> Result<NoteMeta> {
    let mut stmt = conn.prepare(
        "SELECT note_id, title, domain, intent, kind, trust, status, updated_at
         FROM current_notes WHERE note_id = ?1"
    )?;

    let meta = stmt.query_row([note_id], |row| {
        Ok(NoteMeta {
            note_id: row.get(0)?,
            title: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
            domain: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
            intent: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
            kind: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
            trust: row.get::<_, Option<String>>(5)?.unwrap_or_default(),
            status: row.get::<_, Option<String>>(6)?.unwrap_or_default(),
            updated_at: row.get::<_, Option<String>>(7)?.unwrap_or_default(),
            tags: Vec::new(),
        })
    })?;

    let mut tag_stmt = conn.prepare(
        "SELECT t.name FROM tags t
         JOIN note_tags nt ON t.tag_id = nt.tag_id
         WHERE nt.note_id = ?1
         ORDER BY t.name"
    )?;

    let tags: Vec<String> = tag_stmt
        .query_map([note_id], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    Ok(NoteMeta { tags, ..meta })
}

pub fn get_ref(conn: &Connection, note_id: &str) -> Result<NoteRef> {
    let mut stmt = conn.prepare(
        "SELECT nv.fm_hash, nv.md_hash
         FROM notes n
         JOIN note_versions nv ON n.head_version_id = nv.version_id
         WHERE n.note_id = ?1"
    )?;

    let r = stmt.query_row([note_id], |row| {
        Ok(NoteRef {
            fm_hash: row.get(0)?,
            md_hash: row.get(1)?,
        })
    })?;

    Ok(r)
}
