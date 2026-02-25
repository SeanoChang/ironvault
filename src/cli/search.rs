use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::search::{self, SearchFilters};

pub fn run(
    vault_dir: &Path,
    query: &str,
    domain: Option<&str>,
    kind: Option<&str>,
    intent: Option<&str>,
    tags: &[String],
    limit: usize,
) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;

    let filters = SearchFilters {
        domain,
        kind,
        intent,
        tags,
        limit,
    };

    let hits = search::search(&conn, query, &filters)?;

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
        "hits": results.len(),
        "results": results,
    });

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
