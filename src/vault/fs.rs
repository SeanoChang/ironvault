use anyhow::Result;
use serde_yaml;
use serde_json;
use std::path::{Path, PathBuf};
use std::fs;
use std::fs::File;
use std::io::Write;
use uuid;

use crate::types::markdown::Frontmatter;

pub struct Vault {
    root: PathBuf,
}

impl Vault {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn ingest(&self, note: &str, note_id: Option<&str>) -> Result<(String, String)> {
        // parse markdown files
        let (raw_fm, raw_md) = Self::split_frontmatter(note)?;

        // validate frotnmatter
        let fm: Frontmatter = serde_yaml::from_str(raw_fm)?;
        let canonical_fm = serde_yaml::to_string(&fm)?;

        // clean md
        let md: String = raw_md.trim().to_string();

        // store frontmatter and body
        let fm_hash = self.store(&canonical_fm, "objects/fm", "yaml")?;
        let md_hash = self.store(&md, "objects/md", "md")?;
        
        // create or update note with new fm and md hashes
        let (note_id, version_id) = self.create_or_update(&fm_hash, &md_hash, note_id)?;
        
        Ok((note_id, version_id))
    }

    fn split_frontmatter(note: &str) -> Result<(&str, &str)> {
        if note.starts_with("---\n") {
            match note[4..].find("\n---\n") {
                Some(i) => {
                    let fm = &note[4..4+i];
                    let body = &note[4+i+5..];

                    return Ok((fm, body))
                }
                None => {
                    // no closing frontmatter
                    return Err(anyhow::anyhow!("[Vault]: no closing frontmatter"))
                }
            }
        }

        return Err(anyhow::anyhow!("[Vault]: no frontmatter found!"))
    }

    fn store(&self, content: &str, folder_path: &str, file_type: &str) -> Result<String> {
        // hash and get hex string
        let hashed_content = blake3::hash(content.as_bytes());
        let hashed_content_hex = hashed_content.to_hex().to_string();

        // get path
        let path = self.root
            .join(folder_path)
            .join(&hashed_content_hex[0..2])
            .join(format!("{}.{}", hashed_content_hex, file_type));
        
        // write content to path
        self.cas_write(content, &path, &self.root.join("tmp"))?;

        return Ok(hashed_content_hex)
    }

    fn create_or_update(&self, fm_hash: &str, md_hash: &str, note_id: Option<&str>) -> Result<(String, String)> {
        // generate content hash
        let content_hash = blake3::hash(
            format!("{}{}", fm_hash, md_hash).as_bytes()
        ).to_hex().to_string();

        // get note_id and prev_version_id
        let (note_id, prev_version_id) = match note_id {
            Some(id) => {
                let note_path = self.root.join("notes").join(id);
                let head = fs::read_to_string(note_path
                    .join("head"))?.trim().to_string();
                
                (id.to_string(), Some(head))
            }
            None => {
                let id = uuid::Uuid::new_v4().to_string();
                let note_path = self.root.join("notes").join(&id);
                
                fs::create_dir_all(note_path.join("versions"))?;
                
                (id, None)
            }
        };

        let version_id = uuid::Uuid::new_v4().to_string();

        // update head with latest version id
        let head_path = self.root.join(format!("notes/{}/head", note_id));
        fs::write(head_path, &version_id)?;

        // update versions .json and .ref
        // write .ref
        let ref_content = format!(
            r#"{{"fm_hash":"{}","md_hash":"{}"}}"#,
            fm_hash, md_hash
        );
        
        let ref_path = self.root.join("notes").join(&note_id)
            .join("versions").join(format!("{}.ref", version_id));
        
        fs::write(&ref_path, &ref_content)?;

        // write .json
        let json_content = serde_json::json!({
            "version_id": version_id,
            "prev_version_id": prev_version_id,
            "content_hash": content_hash,
            "created_at": chrono::Utc::now().to_rfc3339(),
        }).to_string();
        
        let json_path = self.root.join("notes").join(&note_id)
            .join("versions").join(format!("{}.json", version_id));
        
        fs::write(&json_path, &json_content)?;

        Ok((note_id, version_id))
    }

    fn cas_write(&self, content: &str, final_path: &Path, temp_dir: &Path) -> Result<()> {
        if final_path.exists() {
            // already exists, do nothing
            return Ok(())
        }

        // make sure dir exists
        fs::create_dir_all(final_path.parent().unwrap())?;
        fs::create_dir_all(temp_dir)?;

        // write to temp file
        let temp_path = temp_dir.join(uuid::Uuid::new_v4().to_string());
        let mut f = File::create(&temp_path)?;
        f.write_all(content.as_bytes())?;
        f.sync_all()?;

        // move temp file to final path
        fs::rename(&temp_path, final_path)?;

        Ok(())
    }
}
