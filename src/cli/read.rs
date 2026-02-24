use anyhow::Result;
use std::path::Path;

use crate::db;
use crate::registry::resolve;
use crate::vault::fs::Vault;

pub fn run(vault_dir: &Path, id: &str) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let vault = Vault::new(vault_dir.to_path_buf());

    let meta = resolve::get_meta(&conn, id)?;
    let refs = resolve::get_ref(&conn, id)?;

    let fm_raw = vault.read_object("objects/fm", &refs.fm_hash, "yaml")?;
    let body = vault.read_object("objects/md", &refs.md_hash, "md")?;

    let fm: serde_json::Value = serde_yaml::from_str(&fm_raw)?;

    let out = serde_json::json!({
        "id": meta.note_id,
        "title": meta.title,
        "frontmatter": fm,
        "body": body,
    });

    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}
