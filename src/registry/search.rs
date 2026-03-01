use anyhow::{bail, Result};
use chrono::Utc;
use rusqlite::Connection;

use crate::config::SearchConfig;

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

pub fn search(
    conn: &Connection,
    query: &str,
    filters: &SearchFilters,
    config: &SearchConfig,
) -> Result<Vec<SearchHit>> {
    if let Some(k) = filters.kind
        && !VALID_KINDS.contains(&k)
    {
        bail!("invalid kind \"{}\". Valid kinds: {}", k, VALID_KINDS.join(", "));
    }
    if let Some(i) = filters.intent
        && !VALID_INTENTS.contains(&i)
    {
        bail!("invalid intent \"{}\". Valid intents: {}", i, VALID_INTENTS.join(", "));
    }

    let has_query = !query.is_empty();
    let has_filters = filters.domain.is_some()
        || filters.kind.is_some()
        || filters.intent.is_some()
        || !filters.tags.is_empty();

    if !has_query && !has_filters {
        bail!("search requires a query, --tag, --kind, --intent, --domain, or a combination");
    }

    let candidates = if has_query {
        fetch_fts_candidates(conn, query, filters)?
    } else {
        fetch_filter_candidates(conn, filters)?
    };

    let blended = blend_and_rank(candidates, config);

    let results = blended
        .into_iter()
        .take(filters.limit)
        .map(|c| SearchHit {
            note_id: c.note_id,
            title: c.title,
            domain: c.domain,
            kind: c.kind,
            snippet: c.snippet,
            rank: c.final_score,
        })
        .collect();

    Ok(results)
}

// -- Internal candidate type used during scoring --

struct Candidate {
    note_id: String,
    title: String,
    domain: String,
    kind: String,
    snippet: String,
    bm25_raw: f64,
    importance: i64,
    last_accessed: Option<String>,
    created_at: String,
    final_score: f64,
}

/// Fetch FTS5 candidates with importance + access tracking columns.
fn fetch_fts_candidates(
    conn: &Connection,
    query: &str,
    filters: &SearchFilters,
) -> Result<Vec<Candidate>> {
    let mut sql = String::from(
        "SELECT
            nt.note_id,
            cn.title,
            cn.domain,
            cn.kind,
            snippet(note_text, 2, '[', ']', '...', 32),
            bm25(note_text, 0.0, 5.0, 1.0, 2.0, 3.0, 10.0),
            cn.importance,
            cn.last_accessed,
            cn.updated_at
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

    // Fetch more candidates than limit to allow re-ranking
    let fetch_limit = (filters.limit * 10).max(100);
    sql.push_str(&format!(
        "\n         ORDER BY bm25(note_text, 0.0, 5.0, 1.0, 2.0, 3.0, 10.0)\n         LIMIT ?{}",
        pi
    ));
    params.push(Box::new(fetch_limit as i64));

    exec_candidate_query(conn, &sql, &params)
}

/// Filter-only candidates (no FTS5).
fn fetch_filter_candidates(
    conn: &Connection,
    filters: &SearchFilters,
) -> Result<Vec<Candidate>> {
    let mut sql = String::from(
        "SELECT cn.note_id, cn.title, cn.domain, cn.kind, '' AS snippet, 0.0 AS rank,
                cn.importance, cn.last_accessed, cn.updated_at
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

    let fetch_limit = (filters.limit * 10).max(100);
    sql.push_str(&format!(
        "\n         ORDER BY cn.activation_score DESC, cn.updated_at DESC\n         LIMIT ?{}",
        pi
    ));
    params.push(Box::new(fetch_limit as i64));

    exec_candidate_query(conn, &sql, &params)
}

/// Blend signals and sort by final score.
fn blend_and_rank(mut candidates: Vec<Candidate>, config: &SearchConfig) -> Vec<Candidate> {
    if candidates.is_empty() {
        return candidates;
    }

    let w = &config.weights;
    let lambda = config.recency.lambda;
    let now = Utc::now();

    // Normalize BM25 scores to [0, 1].
    // BM25 from FTS5 returns negative values (more negative = better match).
    let min_bm25 = candidates.iter().map(|c| c.bm25_raw).fold(f64::INFINITY, f64::min);
    let max_bm25 = candidates.iter().map(|c| c.bm25_raw).fold(f64::NEG_INFINITY, f64::max);
    let bm25_range = max_bm25 - min_bm25;

    for c in &mut candidates {
        c.bm25_raw = if bm25_range > 0.0 {
            (max_bm25 - c.bm25_raw) / bm25_range
        } else {
            // Single result, all equal, or all zero (filter-only) → treat as 1.0
            if c.bm25_raw != 0.0 { 1.0 } else { 0.0 }
        };
    }

    for c in &mut candidates {
        let primary = c.bm25_raw; // BM25 fallback (no embeddings yet)
        let importance = c.importance.clamp(0, 10) as f64 / 10.0;

        // Recency: exp(-lambda * days_since_access)
        let reference_time = c.last_accessed.as_deref().unwrap_or(&c.created_at);
        let recency = match chrono::DateTime::parse_from_rfc3339(reference_time) {
            Ok(dt) => {
                let days = (now - dt.with_timezone(&chrono::Utc))
                    .num_seconds() as f64 / 86400.0;
                (-lambda * days.max(0.0)).exp()
            }
            Err(_) => 0.5, // fallback for unparseable timestamps
        };

        // graph = 0.0 for now (no graph expansion in v0)
        let graph = 0.0;

        c.final_score = primary * w.cosine
            + graph * w.graph
            + importance * w.importance
            + recency * w.recency;
    }

    // Sort descending by final_score
    candidates.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal));

    candidates
}

// -- Helpers --

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

fn exec_candidate_query(
    conn: &Connection,
    sql: &str,
    params: &[Box<dyn rusqlite::types::ToSql>],
) -> Result<Vec<Candidate>> {
    let mut stmt = conn.prepare(sql)?;
    let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();

    let rows: Vec<Candidate> = stmt
        .query_map(param_refs.as_slice(), |row| {
            Ok(Candidate {
                note_id: row.get(0)?,
                title: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                domain: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                kind: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                snippet: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                bm25_raw: row.get(5)?,
                importance: row.get::<_, Option<i64>>(6)?.unwrap_or(5),
                last_accessed: row.get::<_, Option<String>>(7)?,
                created_at: row.get::<_, Option<String>>(8)?.unwrap_or_default(),
                final_score: 0.0,
            })
        })?
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
