use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::delete::{self, DeletedNote};
use crate::vault::fs::Vault;

pub fn run(vault_dir: &Path, ids: Vec<String>, force: bool, recursive: bool) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;

    let notes = delete::validate_ids(&conn, &ids)?;

    let mode = if force && recursive {
        purge(&conn, &notes, vault_dir)?;
        "purge"
    } else if force {
        delete::hard_delete(&conn, &notes)?;
        "hard_delete"
    } else {
        delete::soft_delete(&conn, &notes)?;
        "retract"
    };

    let out = serde_json::json!({
        "deleted": notes.len(),
        "mode": mode,
        "notes": notes.iter().map(|n| serde_json::json!({
            "id": n.note_id,
            "title": n.title,
        })).collect::<Vec<_>>(),
    });

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn purge(conn: &rusqlite::Connection, notes: &[DeletedNote], vault_dir: &Path) -> Result<()> {
    delete::hard_delete(conn, notes)?;

    let vault = Vault::new(vault_dir.to_path_buf());
    for note in notes {
        vault.remove_object("objects/fm", &note.fm_hash, "yaml")?;
        vault.remove_object("objects/md", &note.md_hash, "md")?;
    }

    Ok(())
}
