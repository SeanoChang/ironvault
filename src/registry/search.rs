use anyhow::{bail, Result};
use rusqlite::Connection;

pub struct SearchHit {
    pub note_id: String,
    pub title: String,
    pub domain: String,
    pub kind: String,
    pub snippet: String,
    pub rank: f64,
}

pub fn search(
    conn: &Connection,
    query: &str,
    domain: Option<&str>,
    tags: &[String],
    limit: usize,
) -> Result<Vec<SearchHit>> {
    let has_query = !query.is_empty();
    let has_tags = !tags.is_empty();

    if !has_query && !has_tags {
        bail!("search requires a query, --tag filter, or both");
    }

    if has_query {
        search_fts(conn, query, domain, tags, limit)
    } else {
        search_tags_only(conn, domain, tags, limit)
    }
}

/// FTS5 search, optionally filtered by domain and/or tags.
/// Tag filtering uses a subquery to avoid GROUP BY conflicts with FTS5.
fn search_fts(
    conn: &Connection,
    query: &str,
    domain: Option<&str>,
    tags: &[String],
    limit: usize,
) -> Result<Vec<SearchHit>> {
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

    if let Some(d) = domain {
        sql.push_str(&format!("\n           AND cn.domain = ?{}", pi));
        params.push(Box::new(d.to_string()));
        pi += 1;
    }

    if !tags.is_empty() {
        let (subquery, new_pi) = tag_subquery(tags, pi);
        sql.push_str(&format!("\n           AND cn.note_id IN ({})", subquery));
        for tag in tags {
            params.push(Box::new(tag.clone()));
        }
        params.push(Box::new(tags.len() as i64));
        pi = new_pi;
    }

    sql.push_str(&format!(
        "\n         ORDER BY bm25(note_text, 0.0, 5.0, 1.0, 2.0, 3.0, 10.0)\n         LIMIT ?{}",
        pi
    ));
    params.push(Box::new(limit as i64));

    let mut stmt = conn.prepare(&sql)?;
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let rows: Vec<SearchHit> = stmt
        .query_map(param_refs.as_slice(), map_row)?
        .filter_map(|r| r.ok())
        .collect();

    Ok(rows)
}

/// Tag-only search (no FTS5), ordered by recency.
fn search_tags_only(
    conn: &Connection,
    domain: Option<&str>,
    tags: &[String],
    limit: usize,
) -> Result<Vec<SearchHit>> {
    let mut sql = String::from(
        "SELECT cn.note_id, cn.title, cn.domain, cn.kind, '' AS snippet, 0.0 AS rank
         FROM current_notes cn
         WHERE cn.namespace = 'ark'
           AND cn.status != 'retracted'"
    );

    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
    let mut pi = 1usize;

    if let Some(d) = domain {
        sql.push_str(&format!("\n           AND cn.domain = ?{}", pi));
        params.push(Box::new(d.to_string()));
        pi += 1;
    }

    let (subquery, new_pi) = tag_subquery(tags, pi);
    sql.push_str(&format!("\n           AND cn.note_id IN ({})", subquery));
    for tag in tags {
        params.push(Box::new(tag.clone()));
    }
    params.push(Box::new(tags.len() as i64));
    pi = new_pi;

    sql.push_str(&format!("\n         ORDER BY cn.updated_at DESC\n         LIMIT ?{}", pi));
    params.push(Box::new(limit as i64));

    let mut stmt = conn.prepare(&sql)?;
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let rows: Vec<SearchHit> = stmt
        .query_map(param_refs.as_slice(), map_row)?
        .filter_map(|r| r.ok())
        .collect();

    Ok(rows)
}

/// Build a tag-filtering subquery: SELECT note_id FROM note_tags ... HAVING COUNT = N
/// Returns (sql_fragment, next_param_index).
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
