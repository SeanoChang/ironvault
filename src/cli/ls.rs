use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::browse::{self, BrowseResult};

pub fn run(vault_dir: &Path, path: Option<&str>) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let result = browse::browse(&conn, path)?;

    let out = match result {
        BrowseResult::Groups { level, items } => {
            let results: Vec<serde_json::Value> = items.iter().map(|g| {
                serde_json::json!({ "name": g.name, "count": g.count })
            }).collect();
            serde_json::json!({
                "level": level,
                "path": path.unwrap_or(""),
                "results": results,
            })
        }
        BrowseResult::Notes(notes) => {
            let results: Vec<serde_json::Value> = notes.iter().map(|n| {
                serde_json::json!({
                    "id": n.note_id,
                    "title": n.title,
                    "trust": n.trust,
                    "updated_at": n.updated_at,
                })
            }).collect();
            serde_json::json!({
                "level": "note",
                "path": path.unwrap_or(""),
                "results": results,
            })
        }
    };

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
