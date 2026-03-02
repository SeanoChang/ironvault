use anyhow::{bail, Result};
use std::path::Path;

use crate::config;
use crate::db;
use crate::embed;
use crate::registry::{embeddings, search::{self, CosineContext, SearchFilters, SearchMode}};

pub fn run(
    vault_dir: &Path,
    query: &str,
    domain: Option<&str>,
    kind: Option<&str>,
    intent: Option<&str>,
    tags: &[String],
    limit: usize,
    bm25_only: bool,
    semantic: bool,
) -> Result<()> {
    if bm25_only && semantic {
        bail!("--bm25 and --semantic are mutually exclusive");
    }

    let conn = db::open_registry(vault_dir)?;
    let cfg = config::load(vault_dir)?;

    let filters = SearchFilters {
        domain,
        kind,
        intent,
        tags,
        limit,
    };

    let mode = if bm25_only {
        SearchMode::Bm25Only
    } else if semantic {
        SearchMode::Semantic
    } else {
        SearchMode::Normal
    };

    // Build cosine context if embeddings are available and there's a query.
    // Skip for BM25-only mode (doesn't use cosine).
    let cosine_ctx = if mode != SearchMode::Bm25Only && !query.is_empty() && embeddings::has_embeddings(&conn) {
        build_cosine_context(vault_dir, &conn, query)
    } else {
        None
    };

    let hits = search::search(&conn, query, &filters, &cfg.search, cosine_ctx.as_ref(), mode)?;

    let results: Vec<serde_json::Value> = hits.iter().map(|h| {
        serde_json::json!({
            "id": h.note_id,
            "title": h.title,
            "domain": h.domain,
            "kind": h.kind,
            "snippet": h.snippet,
            "rank": h.rank,
        })
    }).collect();

    let out = serde_json::json!({
        "query": query,
        "domain": domain,
        "mode": match mode {
            SearchMode::Bm25Only => "bm25",
            SearchMode::Semantic => "semantic",
            SearchMode::Normal => "normal",
        },
        "hits": results.len(),
        "results": results,
    });

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn build_cosine_context(
    vault_dir: &Path,
    conn: &rusqlite::Connection,
    query: &str,
) -> Option<CosineContext> {
    let mut engine = embed::init_embedding(vault_dir)?;
    let query_embedding = engine.embed_query(query).ok()?;
    let all = embeddings::get_all_embeddings(conn).ok()?;
    let note_embeddings = all.into_iter().collect();
    Some(CosineContext { query_embedding, note_embeddings })
}
