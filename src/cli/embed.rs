use anyhow::{bail, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;

use crate::db;
use crate::embed::{self, build_embed_input, init_embedding};
use crate::registry::{embeddings, resolve};
use crate::vault::fs::Vault;

pub fn run_init(vault_dir: &Path) -> Result<()> {
    embed::download::run_init(vault_dir)
}

pub fn run_build(vault_dir: &Path) -> Result<()> {
    let conn = db::open_registry(vault_dir)?;
    let vault = Vault::new(vault_dir.to_path_buf());

    let mut engine = match init_embedding(vault_dir) {
        Some(e) => e,
        None => bail!("embedding not available. Run `nark embed init` first."),
    };

    let missing = embeddings::get_notes_without_embeddings(&conn)?;
    if missing.is_empty() {
        eprintln!("All notes already embedded.");
        return Ok(());
    }

    eprintln!("Found {} notes without embeddings", missing.len());

    let pb = ProgressBar::new(missing.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("  Embedding... [{bar:40}] {pos}/{len}")
        .unwrap()
        .progress_chars("##-"));

    for note_id in &missing {
        let meta = resolve::get_meta(&conn, note_id)?;
        let refs = resolve::get_ref(&conn, note_id)?;
        let body = vault.read_object("objects/md", &refs.md_hash, "md")?;
        let fm_raw = vault.read_object("objects/fm", &refs.fm_hash, "yaml")?;
        let fm: crate::types::markdown::Frontmatter = serde_yaml::from_str(&fm_raw)?;

        let input = build_embed_input(
            &meta.title, &meta.domain, &meta.kind, &meta.intent,
            &fm.tags, &fm.aliases, &body,
        );

        let embedding = engine.embed_document(&input)?;
        embeddings::upsert_embedding(&conn, note_id, &embedding, "bge-base-en-v1.5")?;

        pb.inc(1);
    }

    pb.finish_and_clear();
    eprintln!("Done. {} notes embedded.", missing.len());

    // Sanity check: cosine self-similarity of first embedded note should ≈ 1.0
    if let Some(first_id) = missing.first() {
        if let Some(vec) = embeddings::get_embedding(&conn, first_id)? {
            let self_sim = embed::cosine_similarity(&vec, &vec);
            if (self_sim - 1.0).abs() > 0.001 {
                eprintln!("Warning: self-similarity sanity check failed ({self_sim:.4} != 1.0)");
            } else {
                eprintln!("Self-similarity sanity check passed ({self_sim:.6}).");
            }
        }
    }

    Ok(())
}
