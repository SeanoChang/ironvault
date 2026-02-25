use anyhow::{bail, Result};
use rusqlite::Connection;

const VALID_KINDS: &[&str] = &[
    "spec", "decision", "runbook", "report", "reference", "incident", "experiment", "dataset",
];
const VALID_INTENTS: &[&str] = &[
    "build", "debug", "operate", "design", "research", "evaluate", "decide",
];

pub struct SearchHit {
    pub note_id: String,
    pub title: String,
    pub domain: String,
    pub kind: String,
    pub snippet: String,
    pub rank: f64,
}

pub struct SearchFilters<'a> {
    pub domain: Option<&'a str>,
    pub kind: Option<&'a str>,
    pub intent: Option<&'a str>,
    pub tags: &'a [String],
    pub limit: usize,
}

pub fn search(conn: &Connection, query: &str, filters: &SearchFilters) -> Result<Vec<SearchHit>> {
    // Validate enum filters
    if let Some(k) = filters.kind {
        if !VALID_KINDS.contains(&k) {
            bail!("invalid kind \"{}\". Valid kinds: {}", k, VALID_KINDS.join(", "));
        }
    }
    if let Some(i) = filters.intent {
        if !VALID_INTENTS.contains(&i) {
            bail!("invalid intent \"{}\". Valid intents: {}", i, VALID_INTENTS.join(", "));
        }
    }

    let has_query = !query.is_empty();
    let has_filters = filters.domain.is_some()
        || filters.kind.is_some()
        || filters.intent.is_some()
        || !filters.tags.is_empty();

    if !has_query && !has_filters {
        bail!("search requires a query, --tag, --kind, --intent, --domain, or a combination");
    }

    if has_query {
        search_fts(conn, query, filters)
    } else {
        search_no_fts(conn, filters)
    }
}

/// FTS5 search with optional column/tag filters.
fn search_fts(conn: &Connection, query: &str, filters: &SearchFilters) -> Result<Vec<SearchHit>> {
    let mut sql = String::from(
        "SELECT
            nt.note_id,
            cn.title,
            cn.domain,
            cn.kind,
            snippet(note_text, 2, '[', ']', '...', 32),
            bm25(note_text, 0.0, 5.0, 1.0, 2.0, 3.0, 10.0)
         FROM note_text nt
         JOIN current_notes cn ON nt.note_id = cn.note_id
         WHERE note_text MATCH ?1
           AND cn.namespace = 'ark'
           AND cn.status != 'retracted'"
    );

    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    params.push(Box::new(query.to_string()));
    let mut pi = 2usize;

    append_column_filters(&mut sql, &mut params, &mut pi, filters);

    if !filters.tags.is_empty() {
        let (subquery, new_pi) = tag_subquery(filters.tags, pi);
        sql.push_str(&format!("\n           AND cn.note_id IN ({})", subquery));
        for tag in filters.tags {
            params.push(Box::new(tag.clone()));
        }
        params.push(Box::new(filters.tags.len() as i64));
        pi = new_pi;
    }

    sql.push_str(&format!(
        "\n         ORDER BY bm25(note_text, 0.0, 5.0, 1.0, 2.0, 3.0, 10.0)\n         LIMIT ?{}",
        pi
    ));
    params.push(Box::new(filters.limit as i64));

    exec_query(conn, &sql, &params)
}

/// Filter-only search (no FTS5), ordered by recency.
fn search_no_fts(conn: &Connection, filters: &SearchFilters) -> Result<Vec<SearchHit>> {
    let mut sql = String::from(
        "SELECT cn.note_id, cn.title, cn.domain, cn.kind, '' AS snippet, 0.0 AS rank
         FROM current_notes cn
         WHERE cn.namespace = 'ark'
           AND cn.status != 'retracted'"
    );

    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    let mut pi = 1usize;

    append_column_filters(&mut sql, &mut params, &mut pi, filters);

    if !filters.tags.is_empty() {
        let (subquery, new_pi) = tag_subquery(filters.tags, pi);
        sql.push_str(&format!("\n           AND cn.note_id IN ({})", subquery));
        for tag in filters.tags {
            params.push(Box::new(tag.clone()));
        }
        params.push(Box::new(filters.tags.len() as i64));
        pi = new_pi;
    }

    sql.push_str(&format!(
        "\n         ORDER BY cn.activation_score DESC, cn.updated_at DESC\n         LIMIT ?{}",
        pi
    ));
    params.push(Box::new(filters.limit as i64));

    exec_query(conn, &sql, &params)
}

/// Append WHERE clauses for domain, kind, intent.
fn append_column_filters(
    sql: &mut String,
    params: &mut Vec<Box<dyn rusqlite::types::ToSql>>,
    pi: &mut usize,
    filters: &SearchFilters,
) {
    if let Some(d) = filters.domain {
        sql.push_str(&format!("\n           AND cn.domain = ?{}", *pi));
        params.push(Box::new(d.to_string()));
        *pi += 1;
    }
    if let Some(k) = filters.kind {
        sql.push_str(&format!("\n           AND cn.kind = ?{}", *pi));
        params.push(Box::new(k.to_string()));
        *pi += 1;
    }
    if let Some(i) = filters.intent {
        sql.push_str(&format!("\n           AND cn.intent = ?{}", *pi));
        params.push(Box::new(i.to_string()));
        *pi += 1;
    }
}

fn exec_query(
    conn: &Connection,
    sql: &str,
    params: &[Box<dyn rusqlite::types::ToSql>],
) -> Result<Vec<SearchHit>> {
    let mut stmt = conn.prepare(sql)?;
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let rows: Vec<SearchHit> = stmt
        .query_map(param_refs.as_slice(), map_row)?
        .filter_map(|r| r.ok())
        .collect();

    Ok(rows)
}

fn tag_subquery(tags: &[String], start_pi: usize) -> (String, usize) {
    let mut pi = start_pi;
    let placeholders: Vec<String> = tags.iter().map(|_| {
        let p = format!("?{}", pi);
        pi += 1;
        p
    }).collect();

    let having_pi = pi;
    pi += 1;

    let sql = format!(
        "SELECT ntg.note_id FROM note_tags ntg \
         JOIN tags t ON t.tag_id = ntg.tag_id \
         WHERE t.name IN ({}) \
         GROUP BY ntg.note_id \
         HAVING COUNT(DISTINCT t.name) = ?{}",
        placeholders.join(", "),
        having_pi
    );

    (sql, pi)
}

fn map_row(row: &rusqlite::Row) -> rusqlite::Result<SearchHit> {
    Ok(SearchHit {
        note_id: row.get(0)?,
        title: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
        domain: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
        kind: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
        snippet: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
        rank: row.get(5)?,
    })
}
