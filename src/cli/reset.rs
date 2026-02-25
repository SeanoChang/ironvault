use anyhow::Result;
use std::path::Path;

use crate::db;

pub fn run(vault_dir: &Path, confirm: bool) -> Result<()> {
    let db_path = vault_dir.join("registry.db");

    if !db_path.exists() {
        let out = serde_json::json!({ "error": "No registry found. Run `nark init` first." });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    let conn = db::open_registry(vault_dir)?;
    let note_count: i64 = conn.query_row("SELECT COUNT(*) FROM notes", [], |r| r.get(0))?;
    let version_count: i64 = conn.query_row("SELECT COUNT(*) FROM note_versions", [], |r| r.get(0))?;
    drop(conn);

    if !confirm {
        let out = serde_json::json!({
            "dry_run": true,
            "notes": note_count,
            "versions": version_count,
            "message": format!(
                "This will destroy the registry ({note_count} notes, {version_count} versions). Vault objects will be kept. Run with --confirm to proceed."
            ),
        });
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(());
    }

    std::fs::remove_file(&db_path)?;
    let _ = std::fs::remove_file(db_path.with_extension("db-wal"));
    let _ = std::fs::remove_file(db_path.with_extension("db-shm"));

    let _conn = db::open_registry(vault_dir)?;

    let out = serde_json::json!({
        "reset": true,
        "destroyed": { "notes": note_count, "versions": version_count },
        "message": "Registry reset. Vault objects untouched.",
    });
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
