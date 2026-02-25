use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::{resolve, search};
use crate::vault::fs::Vault;

pub fn run(vault_dir: &Path, topic: &str, limit: usize) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let vault = Vault::new(vault_dir.to_path_buf());

    let filters = search::SearchFilters {
        domain: None,
        kind: None,
        intent: None,
        tags: &[],
        limit,
    };
    let hits = search::search(&conn, topic, &filters)?;

    let mut results: Vec<serde_json::Value> = Vec::new();
    for hit in &hits {
        let refs = resolve::get_ref(&conn, &hit.note_id)?;
        let body = vault.read_object("objects/md", &refs.md_hash, "md")?;
        let preview = truncate_at_word(&body, 500);

        results.push(serde_json::json!({
            "id": hit.note_id,
            "title": hit.title,
            "domain": hit.domain,
            "kind": hit.kind,
            "body_preview": preview,
        }));
    }

    let out = serde_json::json!({
        "query": topic,
        "results": results,
    });

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn truncate_at_word(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    match s[..max].rfind(' ') {
        Some(i) => &s[..i],
        None => &s[..max],
    }
}
