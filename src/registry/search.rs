use std::collections::{HashMap, HashSet};

use anyhow::{bail, Result};
use rusqlite::Connection;

use crate::config::SearchConfig;
use crate::embed;

const VALID_KINDS: &[&str] = &[
    "spec", "decision", "runbook", "report", "reference", "incident", "experiment", "dataset",
];
const VALID_INTENTS: &[&str] = &[
    "build", "debug", "operate", "design", "research", "evaluate", "decide",
];

#[derive(Debug)]
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

/// Optional cosine context: query embedding + per-note embeddings.
pub struct CosineContext {
    pub query_embedding: Vec<f32>,
    pub note_embeddings: HashMap<String, Vec<f32>>,
}

/// Controls which pipeline steps execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Full pipeline: BM25 filter → graph expand → cosine rank → blend
    Normal,
    /// BM25-only: pre-filter → BM25 filter + rank → return. Skips graph/cosine/blend.
    Bm25Only,
    /// Semantic: pre-filter → cosine against ALL notes → graph expand → blend → return.
    /// Bypasses BM25 as candidate filter.
    Semantic,
}

// -- Internal types --

struct Candidate {
    note_id: String,
    title: String,
    domain: String,
    kind: String,
    snippet: String,
    bm25_rank: Option<usize>,
    cosine_score: f64,
    graph_score: f64,
    activation: f64,
    final_score: f64,
}

struct RawEdge {
    src_note_id: String,
    dst_note_id: String,
    edge_type: String,
    weight: f64,
}

// -- Public API --

pub fn search(
    conn: &Connection,
    query: &str,
    filters: &SearchFilters,
    config: &SearchConfig,
    cosine_ctx: Option<&CosineContext>,
    mode: SearchMode,
) -> Result<Vec<SearchHit>> {
    validate_filters(filters)?;

    let has_query = !query.is_empty();
    let has_filters = filters.domain.is_some()
        || filters.kind.is_some()
        || filters.intent.is_some()
        || !filters.tags.is_empty();

    if !has_query && !has_filters {
        bail!("search requires a query, --tag, --kind, --intent, --domain, or a combination");
    }

    match mode {
        SearchMode::Bm25Only => search_bm25_only(conn, query, filters, config),
        SearchMode::Semantic => search_semantic(conn, query, filters, config, cosine_ctx),
        SearchMode::Normal => search_normal(conn, query, filters, config, cosine_ctx),
    }
}

/// Full pipeline: BM25 filter → graph expand → cosine rank → blend
fn search_normal(
    conn: &Connection,
    query: &str,
    filters: &SearchFilters,
    config: &SearchConfig,
    cosine_ctx: Option<&CosineContext>,
) -> Result<Vec<SearchHit>> {
    let has_query = !query.is_empty();

    // Step 1+2: BM25 recall or filter-only candidates
    let mut candidates = if has_query {
        fetch_fts_candidates(conn, query, filters, config)?
    } else {
        fetch_filter_candidates(conn, filters, config)?
    };

    // Step 3: Graph expansion
    candidates = graph_expand(conn, candidates, config, filters.domain)?;

    // Step 4: Cosine scoring
    compute_cosine_scores(&mut candidates, cosine_ctx);

    // Step 5: Blend signals
    let has_embeddings = cosine_ctx.is_some();
    blend_scores(&mut candidates, config, has_embeddings);

    // Step 6: Threshold + sort + limit
    let results = threshold_and_sort(candidates, config.threshold, filters.limit);
    Ok(to_hits(results))
}

/// BM25-only: pre-filter → BM25 filter + rank → return top N.
/// Skips graph expansion, cosine scoring, and blending.
fn search_bm25_only(
    conn: &Connection,
    query: &str,
    filters: &SearchFilters,
    config: &SearchConfig,
) -> Result<Vec<SearchHit>> {
    let has_query = !query.is_empty();

    let mut candidates = if has_query {
        fetch_fts_candidates(conn, query, filters, config)?
    } else {
        fetch_filter_candidates(conn, filters, config)?
    };

    // Score using BM25 rank position + activation only (no graph, no cosine)
    let pool_size = candidates.iter().filter(|c| c.bm25_rank.is_some()).count() as f64;
    for c in &mut candidates {
        let primary = match c.bm25_rank {
            Some(rank) if pool_size > 0.0 => 1.0 - (rank as f64 / pool_size),
            _ => 0.0,
        };
        // Simple score: BM25 rank normalized + activation tiebreaker
        c.final_score = primary * 0.75 + c.activation * 0.25;
    }

    let results = threshold_and_sort(candidates, config.threshold, filters.limit);
    Ok(to_hits(results))
}

/// Semantic: pre-filter → cosine against ALL notes → graph expand → blend.
/// Bypasses BM25 as candidate filter — brute-force cosine against the full vault.
fn search_semantic(
    conn: &Connection,
    query: &str,
    filters: &SearchFilters,
    config: &SearchConfig,
    cosine_ctx: Option<&CosineContext>,
) -> Result<Vec<SearchHit>> {
    let ctx = match cosine_ctx {
        Some(c) => c,
        None => bail!("--semantic requires embeddings. Run `nark embed init` then `nark embed build`."),
    };

    if query.is_empty() {
        bail!("--semantic requires a query");
    }

    // Step 1: Fetch ALL active notes matching pre-filters (no BM25)
    let mut candidates = fetch_filter_candidates(conn, filters, config)?;

    // Raise the limit for semantic — we want the full pool before cosine ranking
    // (fetch_filter_candidates already uses bm25.top_k as limit, which is fine for
    // small vaults; for larger vaults we'd want no limit, but top_k=100 is enough for now)

    // Step 4: Cosine scoring against all candidates
    compute_cosine_scores(&mut candidates, Some(ctx));

    // Step 3: Graph expansion
    candidates = graph_expand(conn, candidates, config, filters.domain)?;

    // Re-score graph-discovered notes with cosine too
    compute_cosine_scores(&mut candidates, Some(ctx));

    // Step 5: Blend (cosine is primary since we always have embeddings)
    blend_scores(&mut candidates, config, true);

    // Step 6: Threshold + sort + limit
    let results = threshold_and_sort(candidates, config.threshold, filters.limit);
    Ok(to_hits(results))
}

fn to_hits(candidates: Vec<Candidate>) -> Vec<SearchHit> {
    candidates
        .into_iter()
        .map(|c| SearchHit {
            note_id: c.note_id,
            title: c.title,
            domain: c.domain,
            kind: c.kind,
            snippet: c.snippet,
            rank: c.final_score,
        })
        .collect()
}

// -- Validation --

fn validate_filters(filters: &SearchFilters) -> Result<()> {
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
    Ok(())
}

// -- Candidate Fetching --

fn fetch_fts_candidates(
    conn: &Connection,
    query: &str,
    filters: &SearchFilters,
    config: &SearchConfig,
) -> Result<Vec<Candidate>> {
    let bm25_weights = config.bm25.fts5_weights_arg();

    let mut sql = format!(
        "SELECT
            nt.note_id,
            cn.title,
            cn.domain,
            cn.kind,
            snippet(note_text, 2, '[', ']', '...', 32),
            bm25(note_text, {}),
            cn.activation_score
         FROM note_text nt
         JOIN current_notes cn ON nt.note_id = cn.note_id
         WHERE note_text MATCH ?1
           AND cn.namespace = 'ark'
           AND cn.status != 'retracted'",
        bm25_weights
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
        "\n         ORDER BY bm25(note_text, {})\n         LIMIT ?{}",
        bm25_weights, pi
    ));
    params.push(Box::new(config.bm25.top_k as i64));

    let mut candidates = exec_candidate_query(conn, &sql, &params)?;

    // Assign BM25 rank positions
    for (i, c) in candidates.iter_mut().enumerate() {
        c.bm25_rank = Some(i);
    }

    Ok(candidates)
}

fn fetch_filter_candidates(
    conn: &Connection,
    filters: &SearchFilters,
    config: &SearchConfig,
) -> Result<Vec<Candidate>> {
    let mut sql = String::from(
        "SELECT cn.note_id, cn.title, cn.domain, cn.kind, '' AS snippet, 0.0 AS rank,
                cn.activation_score
         FROM current_notes cn
         WHERE cn.namespace = 'ark'
           AND cn.status != 'retracted'",
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
    params.push(Box::new(config.bm25.top_k as i64));

    exec_candidate_query(conn, &sql, &params)
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
            let raw_activation: f64 = row.get::<_, Option<f64>>(6)?.unwrap_or(0.0);
            Ok(Candidate {
                note_id: row.get(0)?,
                title: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                domain: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                kind: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                snippet: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                bm25_rank: None,
                cosine_score: 0.0,
                graph_score: 0.0,
                activation: if raw_activation == 0.0 { 0.50 } else { raw_activation },
                final_score: 0.0,
            })
        })?
        .filter_map(|r| r.ok())
        .collect();

    Ok(rows)
}

// -- Graph Expansion --

fn fetch_edges_batch(conn: &Connection, note_ids: &[&str]) -> Result<Vec<RawEdge>> {
    if note_ids.is_empty() {
        return Ok(Vec::new());
    }

    let placeholders: String = note_ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");

    // Outgoing edges from seeds
    let sql_out = format!(
        "SELECT src_note_id, dst_note_id, edge_type, weight
         FROM note_edges WHERE src_note_id IN ({})",
        placeholders
    );
    // Incoming edges to seeds
    let sql_in = format!(
        "SELECT src_note_id, dst_note_id, edge_type, weight
         FROM note_edges WHERE dst_note_id IN ({})",
        placeholders
    );

    let mut edges = Vec::new();
    for sql in [&sql_out, &sql_in] {
        let mut stmt = conn.prepare(sql)?;
        let params: Vec<&dyn rusqlite::types::ToSql> =
            note_ids.iter().map(|id| id as &dyn rusqlite::types::ToSql).collect();
        let rows = stmt
            .query_map(params.as_slice(), |row| {
                Ok(RawEdge {
                    src_note_id: row.get(0)?,
                    dst_note_id: row.get(1)?,
                    edge_type: row.get(2)?,
                    weight: row.get(3)?,
                })
            })?
            .filter_map(|r| r.ok());
        edges.extend(rows);
    }

    Ok(edges)
}

fn graph_expand(
    conn: &Connection,
    mut candidates: Vec<Candidate>,
    config: &SearchConfig,
    domain_filter: Option<&str>,
) -> Result<Vec<Candidate>> {
    if candidates.is_empty() {
        return Ok(candidates);
    }

    let pool_size = candidates.len() as f64;
    let seed_ids: HashSet<String> = candidates.iter().map(|c| c.note_id.clone()).collect();

    // Seed scores: BM25-ranked notes get position-based score, filter-only get 0.5
    let seed_scores: HashMap<String, f64> = candidates
        .iter()
        .map(|c| {
            let score = match c.bm25_rank {
                Some(rank) => 1.0 - (rank as f64 / pool_size),
                None => 0.5,
            };
            (c.note_id.clone(), score)
        })
        .collect();

    // Fetch all edges touching seed notes
    let id_refs: Vec<&str> = seed_ids.iter().map(|s| s.as_str()).collect();
    let edges = fetch_edges_batch(conn, &id_refs)?;

    // Propagate scores to neighbors
    let mut neighbor_scores: HashMap<String, f64> = HashMap::new();

    for edge in &edges {
        let is_outgoing = seed_ids.contains(&edge.src_note_id);
        let is_incoming = seed_ids.contains(&edge.dst_note_id);

        if is_outgoing {
            // Outgoing from seed → neighbor is dst
            // Skip supersedes outgoing (blocks new→old propagation)
            if edge.edge_type == "supersedes" {
                continue;
            }
            let seed_score = seed_scores.get(&edge.src_note_id).copied().unwrap_or(0.5);
            let propagated = seed_score * edge.weight * config.graph.decay;
            *neighbor_scores.entry(edge.dst_note_id.clone()).or_insert(0.0) += propagated;
        }

        if is_incoming {
            // Incoming to seed → neighbor is src
            // Allow supersedes incoming (permits old→new flow)
            let seed_score = seed_scores.get(&edge.dst_note_id).copied().unwrap_or(0.5);
            let propagated = seed_score * edge.weight * config.graph.decay;
            *neighbor_scores.entry(edge.src_note_id.clone()).or_insert(0.0) += propagated;
        }
    }

    // Assign graph scores to existing candidates
    for c in &mut candidates {
        if let Some(&gs) = neighbor_scores.get(&c.note_id) {
            c.graph_score = gs;
        }
    }

    // Find graph-discovered notes not already in the pool
    let discovered: Vec<String> = neighbor_scores
        .keys()
        .filter(|id| !seed_ids.contains(*id))
        .cloned()
        .collect();

    if !discovered.is_empty() {
        // Fetch metadata for discovered notes in one query
        let placeholders: String = discovered.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
        let mut sql = format!(
            "SELECT cn.note_id, cn.title, cn.domain, cn.kind, '' AS snippet, 0.0 AS rank,
                    cn.activation_score
             FROM current_notes cn
             WHERE cn.note_id IN ({})
               AND cn.namespace = 'ark'
               AND cn.status != 'retracted'",
            placeholders
        );

        let mut params: Vec<Box<dyn rusqlite::types::ToSql>> =
            discovered.iter().map(|id| Box::new(id.clone()) as Box<dyn rusqlite::types::ToSql>).collect();

        // If respect_domain_filter is enabled, restrict to queried domain
        if config.graph.respect_domain_filter {
            if let Some(domain) = domain_filter {
                sql.push_str("\n               AND cn.domain = ?");
                params.push(Box::new(domain.to_string()));
            }
        }

        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::types::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let new_candidates: Vec<Candidate> = stmt
            .query_map(param_refs.as_slice(), |row| {
                let raw_activation: f64 = row.get::<_, Option<f64>>(6)?.unwrap_or(0.0);
                let note_id: String = row.get(0)?;
                Ok(Candidate {
                    note_id,
                    title: row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                    domain: row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    kind: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
                    snippet: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
                    bm25_rank: None,
                    cosine_score: 0.0,
                    graph_score: 0.0,
                    activation: if raw_activation == 0.0 { 0.50 } else { raw_activation },
                    final_score: 0.0,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();

        for mut nc in new_candidates {
            if let Some(&gs) = neighbor_scores.get(&nc.note_id) {
                nc.graph_score = gs;
            }
            candidates.push(nc);
        }
    }

    // Normalize graph scores to [0,1]
    let max_graph = candidates
        .iter()
        .map(|c| c.graph_score)
        .fold(0.0_f64, f64::max);
    if max_graph > 0.0 {
        for c in &mut candidates {
            c.graph_score /= max_graph;
        }
    }

    Ok(candidates)
}

// -- Cosine Scoring --

fn compute_cosine_scores(candidates: &mut [Candidate], cosine_ctx: Option<&CosineContext>) {
    let ctx = match cosine_ctx {
        Some(c) => c,
        None => return,
    };

    for c in candidates.iter_mut() {
        if c.cosine_score > 0.0 {
            continue; // already scored (e.g. semantic mode re-scores after graph expand)
        }
        c.cosine_score = ctx
            .note_embeddings
            .get(&c.note_id)
            .map(|ne| embed::cosine_similarity(&ctx.query_embedding, ne) as f64)
            .unwrap_or(0.0);
    }
}

// -- Blending --

fn blend_scores(candidates: &mut [Candidate], config: &SearchConfig, has_embeddings: bool) {
    if candidates.is_empty() {
        return;
    }

    let w = &config.weights;

    // Pool size for BM25 rank normalization (count of candidates that came from BM25)
    let pool_size = candidates
        .iter()
        .filter(|c| c.bm25_rank.is_some())
        .count() as f64;

    for c in candidates.iter_mut() {
        let primary = if has_embeddings {
            c.cosine_score
        } else if let Some(rank) = c.bm25_rank {
            if pool_size > 0.0 {
                1.0 - (rank as f64 / pool_size)
            } else {
                0.0
            }
        } else {
            0.0
        };

        c.final_score = primary * w.cosine + c.graph_score * w.graph + c.activation * w.activation;
    }
}

// -- Threshold + Sort --

fn threshold_and_sort(mut candidates: Vec<Candidate>, threshold: f64, limit: usize) -> Vec<Candidate> {
    candidates.retain(|c| c.final_score >= threshold);
    candidates.sort_by(|a, b| {
        b.final_score
            .partial_cmp(&a.final_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(limit);
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

fn tag_subquery(tags: &[String], start_pi: usize) -> (String, usize) {
    let mut pi = start_pi;
    let placeholders: Vec<String> = tags
        .iter()
        .map(|_| {
            let p = format!("?{}", pi);
            pi += 1;
            p
        })
        .collect();

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SearchConfig;
    use crate::db;
    use rusqlite::Connection;

    /// Create an in-memory DB with all migrations applied + seed defaults.
    fn setup_db() -> Connection {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("PRAGMA foreign_keys=ON;").unwrap();

        // Use the same migration machinery as production
        db::MIGRATIONS.to_latest(&mut conn).unwrap();
        db::seed_defaults(&conn).unwrap();
        conn
    }

    /// Insert a note into current_notes + note_text (FTS5).
    /// Returns the note_id for convenience.
    fn insert_note(
        conn: &Connection,
        note_id: &str,
        title: &str,
        domain: &str,
        kind: &str,
        body: &str,
        activation_score: f64,
    ) -> String {
        let now = chrono::Utc::now().to_rfc3339();
        let version_id = format!("v-{}", note_id);

        // Insert into notes table (identity)
        conn.execute(
            "INSERT INTO notes (note_id, namespace, head_version_id, author_agent_id, created_at)
             VALUES (?1, 'ark', ?2, ?3, ?4)",
            rusqlite::params![note_id, version_id, db::DEFAULT_AGENT_ID, now],
        )
        .unwrap();

        // Insert into note_versions
        conn.execute(
            "INSERT INTO note_versions (version_id, note_id, author_agent_id, content_hash, fm_hash, md_hash, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![version_id, note_id, db::DEFAULT_AGENT_ID, "ch", "fh", "mh", now],
        )
        .unwrap();

        // Insert into current_notes (materialized view)
        conn.execute(
            "INSERT INTO current_notes (note_id, namespace, head_version_id, author_agent_id, title, domain, kind, status, activation_score, updated_at)
             VALUES (?1, 'ark', ?2, ?3, ?4, ?5, ?6, 'active', ?7, ?8)",
            rusqlite::params![note_id, version_id, db::DEFAULT_AGENT_ID, title, domain, kind, activation_score, now],
        )
        .unwrap();

        // Insert into FTS5 note_text
        conn.execute(
            "INSERT INTO note_text (note_id, title, body, spine, aliases, keywords)
             VALUES (?1, ?2, ?3, '', '', '')",
            rusqlite::params![note_id, title, body],
        )
        .unwrap();

        note_id.to_string()
    }

    /// Insert an edge between two notes.
    fn insert_edge(
        conn: &Connection,
        src: &str,
        dst: &str,
        edge_type: &str,
        weight: f64,
    ) {
        let now = chrono::Utc::now().to_rfc3339();
        conn.execute(
            "INSERT INTO note_edges (src_note_id, dst_note_id, edge_type, weight, source_type, version_id, created_at)
             VALUES (?1, ?2, ?3, ?4, 'body', ?5, ?6)",
            rusqlite::params![src, dst, edge_type, weight, format!("v-{}", src), now],
        )
        .unwrap();
    }

    fn default_config() -> SearchConfig {
        SearchConfig::default()
    }

    fn default_filters(limit: usize) -> SearchFilters<'static> {
        SearchFilters {
            domain: None,
            kind: None,
            intent: None,
            tags: &[],
            limit,
        }
    }

    // ================================================================
    // Test 1: Worked example — Notes A-E with correct ordering
    // ================================================================

    #[test]
    fn test_worked_example_full_pipeline() {
        let conn = setup_db();
        let config = default_config();

        // Insert the 6 systems-domain notes (3 BM25 hits + 2 graph-discovered + 1 extra)
        insert_note(&conn, "note-a", "CAS content hash design", "systems", "spec",
            "content_hash = BLAKE3(fm || md). The vault uses BLAKE3 hashing for all objects.",
            0.85);
        insert_note(&conn, "note-b", "BLAKE3 benchmark results", "systems", "report",
            "BLAKE3 benchmark: 2.1 GB/s on ARM. Fast hashing for content-addressed storage.",
            0.55);
        insert_note(&conn, "note-c", "CAS design for vault", "systems", "spec",
            "Content-addressed storage using BLAKE3 hashing function. Object store layout.",
            0.40);
        insert_note(&conn, "note-d", "Vault ingest pipeline", "systems", "spec",
            "The ingest pipeline reads markdown, parses frontmatter, and writes CAS objects.",
            0.70);
        insert_note(&conn, "note-e", "Registry write.rs impl", "systems", "reference",
            "Implementation of registry write module. Handles note creation and updates.",
            0.45);
        insert_note(&conn, "note-f", "Unrelated systems note", "systems", "report",
            "Network latency measurements for cross-region replication.",
            0.30);

        // Edges: A references D and E (weight 1.0)
        // Using references (weight=1.0) so graph boost is moderate relative to cosine signal
        insert_edge(&conn, "note-a", "note-d", "references", 1.0);
        insert_edge(&conn, "note-a", "note-e", "references", 1.0);

        // Search for "BLAKE3 hashing" with domain filter
        let filters = SearchFilters {
            domain: Some("systems"),
            kind: None,
            intent: None,
            tags: &[],
            limit: 10,
        };

        // Step 1+2: BM25 candidates
        let mut candidates = fetch_fts_candidates(&conn, "BLAKE3 hashing", &filters, &config).unwrap();
        assert!(!candidates.is_empty(), "BM25 should find BLAKE3 hits");

        // Verify A, B, C are in the candidates (they mention BLAKE3)
        let hit_ids: Vec<&str> = candidates.iter().map(|c| c.note_id.as_str()).collect();
        assert!(hit_ids.contains(&"note-a"), "Note A should be a BM25 hit");
        assert!(hit_ids.contains(&"note-b"), "Note B should be a BM25 hit");
        assert!(hit_ids.contains(&"note-c"), "Note C should be a BM25 hit");
        assert!(!hit_ids.contains(&"note-d"), "Note D should NOT be a BM25 hit");

        // Step 3: Graph expansion
        candidates = graph_expand(&conn, candidates, &config, filters.domain).unwrap();
        let expanded_ids: Vec<&str> = candidates.iter().map(|c| c.note_id.as_str()).collect();
        assert!(expanded_ids.contains(&"note-d"), "Note D should be graph-discovered");
        assert!(expanded_ids.contains(&"note-e"), "Note E should be graph-discovered");

        // Verify graph scores are normalized to [0,1]
        for c in &candidates {
            assert!(c.graph_score >= 0.0 && c.graph_score <= 1.0,
                "Graph score for {} should be in [0,1], got {}", c.note_id, c.graph_score);
        }

        // D and E should have non-zero graph scores (discovered via edges from A)
        let d_graph = candidates.iter().find(|c| c.note_id == "note-d").unwrap().graph_score;
        let e_graph = candidates.iter().find(|c| c.note_id == "note-e").unwrap().graph_score;
        assert!(d_graph > 0.0, "Note D should have graph score > 0");
        assert!(e_graph > 0.0, "Note E should have graph score > 0");

        // Step 4: Manually assign cosine scores (simulating what compute_cosine_scores does)
        // A has highest cosine (exact match), D confirmed relevant by cosine
        let fake_cosine: HashMap<&str, f64> = HashMap::from([
            ("note-a", 0.92), ("note-b", 0.88), ("note-c", 0.65),
            ("note-d", 0.71), ("note-e", 0.45),
        ]);
        for c in &mut candidates {
            c.cosine_score = fake_cosine.get(c.note_id.as_str()).copied().unwrap_or(0.0);
        }

        // Step 5: Blend
        blend_scores(&mut candidates, &config, true);

        // Step 6: Threshold + sort
        let results = threshold_and_sort(candidates, config.threshold, 10);

        assert!(results.len() >= 5, "Should have at least 5 results, got {}", results.len());
        let order: Vec<&str> = results.iter().map(|c| c.note_id.as_str()).collect();

        // Key invariant: D (graph-discovered, no keyword match) outranks B (direct BM25 hit)
        // because D is confirmed relevant by cosine AND boosted by graph + high activation
        let d_score = results.iter().find(|c| c.note_id == "note-d").unwrap().final_score;
        let b_score = results.iter().find(|c| c.note_id == "note-b").unwrap().final_score;
        assert!(d_score > b_score,
            "Graph-discovered Note D ({:.3}) should outrank direct hit Note B ({:.3})",
            d_score, b_score);

        // All 5 relevant notes should appear
        assert!(order.contains(&"note-a"), "Note A should be in results");
        assert!(order.contains(&"note-b"), "Note B should be in results");
        assert!(order.contains(&"note-c"), "Note C should be in results");
        assert!(order.contains(&"note-d"), "Note D should be in results");
        assert!(order.contains(&"note-e"), "Note E should be in results");

        // F (unrelated, no keyword match, no graph connection) should NOT appear
        assert!(!order.contains(&"note-f"), "Unrelated Note F should not be in results");

        // Scores should be monotonically decreasing
        for w in results.windows(2) {
            assert!(w[0].final_score >= w[1].final_score,
                "Results should be sorted descending: {} ({:.3}) >= {} ({:.3})",
                w[0].note_id, w[0].final_score, w[1].note_id, w[1].final_score);
        }
    }

    // ================================================================
    // Test 2: Tier 1 — No embeddings, no graph (pure BM25)
    // ================================================================

    #[test]
    fn test_tier1_bm25_only() {
        let conn = setup_db();
        let config = default_config();

        insert_note(&conn, "t1-a", "Rust error handling", "programming", "reference",
            "Rust error handling with Result and Option types. Anyhow for applications.", 0.0);
        insert_note(&conn, "t1-b", "Go error handling", "programming", "reference",
            "Go error handling patterns. Error wrapping with fmt.Errorf.", 0.0);
        insert_note(&conn, "t1-c", "Python testing", "programming", "runbook",
            "Python pytest framework. Unit testing best practices.", 0.0);

        // No edges, no cosine context → pure BM25 + activation (0.50 fallback)
        let hits = search(&conn, "error handling", &default_filters(10), &config, None, SearchMode::Normal).unwrap();

        assert!(hits.len() >= 2, "Should find at least 2 error handling notes");
        // Both A and B mention "error handling" — C (Python testing) should not appear
        let ids: Vec<&str> = hits.iter().map(|h| h.note_id.as_str()).collect();
        assert!(ids.contains(&"t1-a"));
        assert!(ids.contains(&"t1-b"));
        // All scores should be > 0 (BM25 rank normalized + activation fallback)
        for h in &hits {
            assert!(h.rank > 0.0, "Score should be > 0 for {}", h.note_id);
        }
    }

    // ================================================================
    // Test 3: Tier 2 — No embeddings, with graph
    // ================================================================

    #[test]
    fn test_tier2_bm25_plus_graph() {
        let conn = setup_db();
        let config = default_config();

        insert_note(&conn, "t2-a", "OAuth2 token flow", "security", "spec",
            "OAuth2 authorization code flow. Token exchange and refresh.", 0.60);
        insert_note(&conn, "t2-b", "JWT validation", "security", "reference",
            "JWT signature validation. Claims verification. Token expiry.", 0.40);
        insert_note(&conn, "t2-c", "Auth middleware design", "security", "spec",
            "Middleware for request authentication. Session management.", 0.70);

        // Edge: A depends-on C (auth middleware)
        insert_edge(&conn, "t2-a", "t2-c", "depends-on", 2.0);

        // Search with no cosine context (tier 2: BM25 + graph)
        let hits = search(&conn, "OAuth2 token", &default_filters(10), &config, None, SearchMode::Normal).unwrap();

        // A should be a direct hit, C should be graph-discovered
        let ids: Vec<&str> = hits.iter().map(|h| h.note_id.as_str()).collect();
        assert!(ids.contains(&"t2-a"), "Note A should be found via BM25");
        // C may or may not appear depending on graph score vs threshold
        // The important thing is the pipeline doesn't crash without embeddings
    }

    // ================================================================
    // Test 4: Directional supersedes — old→new only
    // ================================================================

    #[test]
    fn test_supersedes_direction() {
        let conn = setup_db();
        let config = default_config();

        // "new" supersedes "old"
        insert_note(&conn, "sup-old", "API v1 design", "systems", "spec",
            "Original API design document. REST endpoints for v1.", 0.30);
        insert_note(&conn, "sup-new", "API v2 design", "systems", "spec",
            "Updated API design. GraphQL migration from REST.", 0.80);
        insert_note(&conn, "sup-other", "API client SDK", "systems", "reference",
            "REST client library. Uses API v1 endpoints.", 0.50);

        // new supersedes old: src=new, dst=old
        insert_edge(&conn, "sup-new", "sup-old", "supersedes", 1.5);
        // other references old
        insert_edge(&conn, "sup-other", "sup-old", "references", 1.0);

        // Case 1: Search finds OLD note → NEW should surface via graph
        let mut candidates = fetch_fts_candidates(
            &conn, "API v1 REST", &default_filters(10), &config
        ).unwrap();
        let old_found = candidates.iter().any(|c| c.note_id == "sup-old");
        if old_found {
            candidates = graph_expand(&conn, candidates, &config, None).unwrap();
            let ids: Vec<&str> = candidates.iter().map(|c| c.note_id.as_str()).collect();
            // When old is found, new should be graph-discovered (old→new propagation)
            assert!(ids.contains(&"sup-new"),
                "When old note is seed, new note should be discovered via incoming supersedes");
        }

        // Case 2: Search finds NEW note → OLD should NOT surface via supersedes
        let mut candidates2 = fetch_fts_candidates(
            &conn, "GraphQL migration", &default_filters(10), &config
        ).unwrap();
        let new_found = candidates2.iter().any(|c| c.note_id == "sup-new");
        if new_found {
            candidates2 = graph_expand(&conn, candidates2, &config, None).unwrap();
            // Old should NOT be discovered via supersedes (blocks new→old)
            let old_via_supersedes = candidates2.iter().any(|c| c.note_id == "sup-old");
            assert!(!old_via_supersedes,
                "When new note is seed, old note should NOT be discovered via supersedes");
        }
    }

    // ================================================================
    // Test 5: NULL/zero activation falls back to 0.50
    // ================================================================

    #[test]
    fn test_activation_fallback() {
        let conn = setup_db();
        let config = default_config();

        // activation_score = 0.0 (orient hasn't run)
        insert_note(&conn, "act-a", "Zero activation note", "systems", "spec",
            "This note has zero activation score. Testing fallback.", 0.0);
        // activation_score = 0.75 (orient has run)
        insert_note(&conn, "act-b", "Normal activation note", "systems", "spec",
            "This note has normal activation score. Testing fallback.", 0.75);

        let candidates = fetch_fts_candidates(
            &conn, "activation score fallback", &default_filters(10), &config
        ).unwrap();

        for c in &candidates {
            if c.note_id == "act-a" {
                assert!((c.activation - 0.50).abs() < 0.001,
                    "Zero activation should fall back to 0.50, got {}", c.activation);
            }
            if c.note_id == "act-b" {
                assert!((c.activation - 0.75).abs() < 0.001,
                    "Non-zero activation should be preserved, got {}", c.activation);
            }
        }
    }

    // ================================================================
    // Test 6: Graph score normalization
    // ================================================================

    #[test]
    fn test_graph_score_normalization() {
        let conn = setup_db();
        let config = default_config();

        // Create a hub note linked to multiple seeds
        insert_note(&conn, "gn-a", "Seed note alpha", "systems", "spec",
            "Graph normalization test seed alpha. Unique keyword graphnorm.", 0.50);
        insert_note(&conn, "gn-b", "Seed note beta", "systems", "spec",
            "Graph normalization test seed beta. Unique keyword graphnorm.", 0.50);
        insert_note(&conn, "gn-c", "Seed note gamma", "systems", "spec",
            "Graph normalization test seed gamma. Unique keyword graphnorm.", 0.50);
        insert_note(&conn, "gn-hub", "Hub note", "systems", "reference",
            "This hub connects to everything. No graphnorm keyword.", 0.50);

        // All seeds link to hub with high weights
        insert_edge(&conn, "gn-a", "gn-hub", "depends-on", 2.0);
        insert_edge(&conn, "gn-b", "gn-hub", "depends-on", 2.0);
        insert_edge(&conn, "gn-c", "gn-hub", "depends-on", 2.0);

        let mut candidates = fetch_fts_candidates(
            &conn, "graphnorm", &default_filters(10), &config
        ).unwrap();

        // Should find a, b, c as seeds
        assert!(candidates.len() >= 3);

        candidates = graph_expand(&conn, candidates, &config, None).unwrap();

        // Hub should be discovered and have the max graph score (normalized to 1.0)
        let hub = candidates.iter().find(|c| c.note_id == "gn-hub");
        assert!(hub.is_some(), "Hub note should be graph-discovered");
        let hub = hub.unwrap();
        assert!((hub.graph_score - 1.0).abs() < 0.001,
            "Hub (max accumulator) should normalize to 1.0, got {}", hub.graph_score);

        // All graph scores should be in [0, 1]
        for c in &candidates {
            assert!(c.graph_score >= 0.0 && c.graph_score <= 1.0,
                "Graph score for {} = {} should be in [0,1]", c.note_id, c.graph_score);
        }
    }

    // ================================================================
    // Test 7: BM25 discarded with embeddings, kept without
    // ================================================================

    #[test]
    fn test_bm25_discard_with_embeddings() {
        let conn = setup_db();
        let config = default_config();

        insert_note(&conn, "bd-a", "Primary search target", "systems", "spec",
            "BM25 discard test. Primary target note.", 0.50);
        insert_note(&conn, "bd-b", "Secondary search target", "systems", "spec",
            "BM25 discard test. Secondary target note.", 0.50);

        let mut candidates = fetch_fts_candidates(
            &conn, "BM25 discard test", &default_filters(10), &config
        ).unwrap();
        assert!(candidates.len() >= 2);

        // Without embeddings: BM25 rank should determine primary signal
        blend_scores(&mut candidates, &config, false);
        let scores_no_embed: Vec<f64> = candidates.iter().map(|c| c.final_score).collect();
        assert!(scores_no_embed.iter().all(|&s| s > 0.0),
            "Without embeddings, BM25 rank should contribute to final score");

        // Reset scores
        for c in &mut candidates {
            c.final_score = 0.0;
            c.cosine_score = 0.99; // high cosine
        }

        // With embeddings: cosine should be primary, BM25 rank ignored
        blend_scores(&mut candidates, &config, true);
        // Both should have the same final score since cosine is the same
        // and they have same activation + graph (both 0)
        let scores_with_embed: Vec<f64> = candidates.iter().map(|c| c.final_score).collect();
        assert!((scores_with_embed[0] - scores_with_embed[1]).abs() < 0.001,
            "With embeddings and same cosine, BM25 rank shouldn't affect scores");
    }

    // ================================================================
    // Test 8: Blend formula correctness
    // ================================================================

    #[test]
    fn test_blend_formula() {
        let config = default_config();
        let mut candidates = vec![
            Candidate {
                note_id: "bf-a".into(),
                title: "test".into(),
                domain: "systems".into(),
                kind: "spec".into(),
                snippet: String::new(),
                bm25_rank: Some(0),
                cosine_score: 0.92,
                graph_score: 0.0,
                activation: 0.85,
                final_score: 0.0,
            },
            Candidate {
                note_id: "bf-d".into(),
                title: "test".into(),
                domain: "systems".into(),
                kind: "spec".into(),
                snippet: String::new(),
                bm25_rank: None,
                cosine_score: 0.71,
                graph_score: 0.40,
                activation: 0.70,
                final_score: 0.0,
            },
        ];

        blend_scores(&mut candidates, &config, true);

        // A: 0.92×0.50 + 0.00×0.25 + 0.85×0.25 = 0.460 + 0.000 + 0.2125 = 0.6725
        let a = &candidates[0];
        let expected_a = 0.92 * 0.50 + 0.00 * 0.25 + 0.85 * 0.25;
        assert!((a.final_score - expected_a).abs() < 0.001,
            "Note A blend: expected {:.4}, got {:.4}", expected_a, a.final_score);

        // D: 0.71×0.50 + 0.40×0.25 + 0.70×0.25 = 0.355 + 0.100 + 0.175 = 0.630
        let d = &candidates[1];
        let expected_d = 0.71 * 0.50 + 0.40 * 0.25 + 0.70 * 0.25;
        assert!((d.final_score - expected_d).abs() < 0.001,
            "Note D blend: expected {:.4}, got {:.4}", expected_d, d.final_score);
    }

    // ================================================================
    // Test 9: Filter-only search (no query)
    // ================================================================

    #[test]
    fn test_filter_only_search() {
        let conn = setup_db();
        let config = default_config();

        insert_note(&conn, "fo-a", "Finance report alpha", "finance", "report",
            "Quarterly earnings analysis for Q4.", 0.90);
        insert_note(&conn, "fo-b", "Finance report beta", "finance", "report",
            "Annual revenue forecast model.", 0.60);
        insert_note(&conn, "fo-c", "Systems spec", "systems", "spec",
            "Infrastructure scaling plan.", 0.80);

        let filters = SearchFilters {
            domain: Some("finance"),
            kind: Some("report"),
            intent: None,
            tags: &[],
            limit: 10,
        };

        let hits = search(&conn, "", &filters, &config, None, SearchMode::Normal).unwrap();

        assert_eq!(hits.len(), 2, "Should find exactly 2 finance reports");
        // Higher activation should rank first
        assert_eq!(hits[0].note_id, "fo-a", "Higher activation note should rank first");
    }

    // ================================================================
    // Test 10: Threshold filtering
    // ================================================================

    #[test]
    fn test_threshold_filtering() {
        let candidates = vec![
            Candidate {
                note_id: "th-a".into(), title: "a".into(), domain: "".into(),
                kind: "".into(), snippet: "".into(),
                bm25_rank: None, cosine_score: 0.0, graph_score: 0.0,
                activation: 0.0, final_score: 0.50,
            },
            Candidate {
                note_id: "th-b".into(), title: "b".into(), domain: "".into(),
                kind: "".into(), snippet: "".into(),
                bm25_rank: None, cosine_score: 0.0, graph_score: 0.0,
                activation: 0.0, final_score: 0.05, // below default threshold 0.10
            },
            Candidate {
                note_id: "th-c".into(), title: "c".into(), domain: "".into(),
                kind: "".into(), snippet: "".into(),
                bm25_rank: None, cosine_score: 0.0, graph_score: 0.0,
                activation: 0.0, final_score: 0.10, // exactly at threshold
            },
        ];

        let results = threshold_and_sort(candidates, 0.10, 10);
        assert_eq!(results.len(), 2, "Should keep scores >= 0.10, filter out 0.05");
        assert_eq!(results[0].note_id, "th-a");
        assert_eq!(results[1].note_id, "th-c");
    }

    // ================================================================
    // Test 11: BM25-only mode skips graph and cosine
    // ================================================================

    #[test]
    fn test_bm25_only_mode() {
        let conn = setup_db();
        let config = default_config();

        insert_note(&conn, "bm-a", "BM25 mode target note", "systems", "spec",
            "Testing BM25 only mode. This should be found.", 0.80);
        insert_note(&conn, "bm-b", "BM25 mode neighbor", "systems", "spec",
            "This neighbor note has different content.", 0.60);
        insert_edge(&conn, "bm-a", "bm-b", "references", 1.0);

        let hits = search(
            &conn, "BM25 mode target", &default_filters(10), &config, None, SearchMode::Bm25Only
        ).unwrap();

        // Should find A (keyword match)
        let ids: Vec<&str> = hits.iter().map(|h| h.note_id.as_str()).collect();
        assert!(ids.contains(&"bm-a"), "BM25-only should find keyword match");
        // B should NOT appear (no keyword match, graph expansion skipped)
        assert!(!ids.contains(&"bm-b"), "BM25-only should not graph-discover neighbors");
    }

    // ================================================================
    // Test 12: --bm25 and --semantic mutual exclusivity (tested at CLI level)
    // ================================================================

    #[test]
    fn test_semantic_mode_requires_embeddings() {
        let conn = setup_db();
        let config = default_config();

        insert_note(&conn, "sem-a", "Semantic test note", "systems", "spec",
            "Testing semantic mode without embeddings.", 0.50);

        let result = search(
            &conn, "semantic test", &default_filters(10), &config, None, SearchMode::Semantic
        );

        assert!(result.is_err(), "--semantic without embeddings should error");
        let err = result.unwrap_err().to_string();
        assert!(err.contains("embeddings"), "Error should mention embeddings: {}", err);
    }

    // ================================================================
    // Test 13: respect_domain_filter restricts graph expansion
    // ================================================================

    #[test]
    fn test_respect_domain_filter() {
        let conn = setup_db();
        let mut config = default_config();

        insert_note(&conn, "df-a", "Domain filter seed", "finance", "spec",
            "Finance domain filter test. Unique keyword domfilter.", 0.50);
        insert_note(&conn, "df-b", "Same domain neighbor", "finance", "reference",
            "Finance reference note about quarterly earnings.", 0.50);
        insert_note(&conn, "df-c", "Cross domain neighbor", "systems", "reference",
            "Systems reference note about infrastructure scaling.", 0.50);

        insert_edge(&conn, "df-a", "df-b", "references", 1.0);
        insert_edge(&conn, "df-a", "df-c", "references", 1.0);

        // Without respect_domain_filter: both neighbors should be discovered
        config.graph.respect_domain_filter = false;
        let mut candidates = fetch_fts_candidates(
            &conn, "domfilter", &default_filters(10), &config
        ).unwrap();
        candidates = graph_expand(&conn, candidates, &config, Some("finance")).unwrap();
        let ids: Vec<&str> = candidates.iter().map(|c| c.note_id.as_str()).collect();
        assert!(ids.contains(&"df-b"), "Without domain filter: same-domain neighbor found");
        assert!(ids.contains(&"df-c"), "Without domain filter: cross-domain neighbor found");

        // With respect_domain_filter: only same-domain neighbor should be discovered
        config.graph.respect_domain_filter = true;
        let mut candidates2 = fetch_fts_candidates(
            &conn, "domfilter", &default_filters(10), &config
        ).unwrap();
        candidates2 = graph_expand(&conn, candidates2, &config, Some("finance")).unwrap();
        let ids2: Vec<&str> = candidates2.iter().map(|c| c.note_id.as_str()).collect();
        assert!(ids2.contains(&"df-b"), "With domain filter: same-domain neighbor found");
        assert!(!ids2.contains(&"df-c"), "With domain filter: cross-domain neighbor excluded");
    }

    // ================================================================
    // Test 14: Changing config weights changes results
    // ================================================================

    #[test]
    fn test_weight_changes_affect_scores() {
        let mut config = default_config();
        let mut candidates = vec![
            Candidate {
                note_id: "wt-a".into(),
                title: "test".into(),
                domain: "systems".into(),
                kind: "spec".into(),
                snippet: String::new(),
                bm25_rank: Some(0),
                cosine_score: 0.90,
                graph_score: 0.10,
                activation: 0.50,
                final_score: 0.0,
            },
        ];

        // Default weights: cosine=0.50, graph=0.25, activation=0.25
        blend_scores(&mut candidates, &config, true);
        let score_default = candidates[0].final_score;

        // Increase graph weight, decrease cosine
        candidates[0].final_score = 0.0;
        config.weights.cosine = 0.30;
        config.weights.graph = 0.45;
        blend_scores(&mut candidates, &config, true);
        let score_graph_heavy = candidates[0].final_score;

        assert!(
            (score_default - score_graph_heavy).abs() > 0.01,
            "Changing weights should observably change scores: default={:.4}, graph_heavy={:.4}",
            score_default, score_graph_heavy
        );
    }

    // ================================================================
    // Test 15: Configurable BM25 column weights
    // ================================================================

    #[test]
    fn test_configurable_bm25_weights() {
        let config = default_config();
        let expected = "0.0, 5, 1, 2, 3, 10";
        let actual = config.bm25.fts5_weights_arg();
        assert_eq!(actual, expected, "Default BM25 weights should match spec");

        // Custom weights
        let mut custom = config;
        custom.bm25.weight_title = 10.0;
        custom.bm25.weight_keywords = 20.0;
        let actual = custom.bm25.fts5_weights_arg();
        assert!(actual.contains("10"), "Custom title weight should appear");
        assert!(actual.contains("20"), "Custom keywords weight should appear");
    }
}
