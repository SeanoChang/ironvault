mod types;
mod vault;

use vault::fs::Vault;
use std::fs;

fn main() {
    let vault = Vault::new("./.vault".into());

    let test = fs::read_to_string("./temp.md").unwrap();
    let (note_id, version_id) = vault.ingest(&test, Some("cf58557c-942b-4581-8419-40a18fb79c4b")).unwrap();
}
