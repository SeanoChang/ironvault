use anyhow::Result;
use rusqlite::Connection;

const FILTER: &str = "namespace = 'ark' AND status != 'deprecated'";

pub struct GroupRow {
    pub name: String,
    pub count: i64,
}

pub struct NoteRow {
    pub note_id: String,
    pub title: String,
    pub trust: String,
    pub updated_at: String,
}

pub enum BrowseResult {
    Groups { level: &'static str, items: Vec<GroupRow> },
    Notes(Vec<NoteRow>),
}

pub fn browse(conn: &Connection, path: Option<&str>) -> Result<BrowseResult> {
    let segments = parse_path(path);

    match segments.as_slice() {
        [] => list_domains(conn),
        [domain] => list_intents(conn, domain),
        [domain, intent] => list_kinds(conn, domain, intent),
        [domain, intent, kind] => list_notes(conn, domain, intent, kind),
        _ => anyhow::bail!("too many path segments (max 3: domain/intent/kind)"),
    }
}

fn parse_path(path: Option<&str>) -> Vec<String> {
    match path {
        None => Vec::new(),
        Some(p) => p.trim_matches('/')
            .split('/')
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect(),
    }
}

fn list_domains(conn: &Connection) -> Result<BrowseResult> {
    let sql = format!(
        "SELECT domain, COUNT(*) FROM current_notes WHERE {} GROUP BY domain ORDER BY domain",
        FILTER
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([], |row| {
        Ok(GroupRow { name: row.get(0)?, count: row.get(1)? })
    })?;
    let items: Vec<GroupRow> = rows.filter_map(|r| r.ok()).collect();
    Ok(BrowseResult::Groups { level: "domain", items })
}

fn list_intents(conn: &Connection, domain: &str) -> Result<BrowseResult> {
    let sql = format!(
        "SELECT intent, COUNT(*) FROM current_notes WHERE {} AND domain = ?1 GROUP BY intent ORDER BY intent",
        FILTER
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map([domain], |row| {
        Ok(GroupRow { name: row.get(0)?, count: row.get(1)? })
    })?;
    let items: Vec<GroupRow> = rows.filter_map(|r| r.ok()).collect();
    Ok(BrowseResult::Groups { level: "intent", items })
}

fn list_kinds(conn: &Connection, domain: &str, intent: &str) -> Result<BrowseResult> {
    let sql = format!(
        "SELECT kind, COUNT(*) FROM current_notes WHERE {} AND domain = ?1 AND intent = ?2 GROUP BY kind ORDER BY kind",
        FILTER
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params![domain, intent], |row| {
        Ok(GroupRow { name: row.get(0)?, count: row.get(1)? })
    })?;
    let items: Vec<GroupRow> = rows.filter_map(|r| r.ok()).collect();
    Ok(BrowseResult::Groups { level: "kind", items })
}

fn list_notes(conn: &Connection, domain: &str, intent: &str, kind: &str) -> Result<BrowseResult> {
    let sql = format!(
        "SELECT note_id, title, trust, updated_at FROM current_notes
         WHERE {} AND domain = ?1 AND intent = ?2 AND kind = ?3
         ORDER BY activation_score DESC, updated_at DESC
         LIMIT 50",
        FILTER
    );
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params![domain, intent, kind], |row| {
        Ok(NoteRow {
            note_id: row.get(0)?,
            title: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
            trust: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
            updated_at: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
        })
    })?;
    let items: Vec<NoteRow> = rows.filter_map(|r| r.ok()).collect();
    Ok(BrowseResult::Notes(items))
}
