use rusqlite::Connection;
use rusqlite_migration::Migrations;
use std::path::Path;
use anyhow::Result;
use std::sync::LazyLock;
use include_dir::{include_dir, Dir};

static MIGRATIONS_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/migrations");

static MIGRATIONS: LazyLock<Migrations<'static>> = LazyLock::new(|| {
    Migrations::from_directory(&MIGRATIONS_DIR).unwrap()
});

pub fn open_registry(vault_dir: &Path) -> Result<Connection> {
    let db_path = vault_dir.join("registry.db");
    let conn = Connection::open(db_path)?;

    conn.execute_batch(
        "PRAGMA journal_mode=WAL;
         PRAGMA foreign_keys=ON;"
    )?;

    Ok(conn)
}

pub fn migrate(conn: &mut Connection) -> Result<(), rusqlite_migration::Error> {
    MIGRATIONS.to_latest(conn)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn test_migration() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&mut conn).unwrap();

        // Check if tables exist
        let mut stmt = conn.prepare("SELECT name FROM sqlite_master WHERE type='table'").unwrap();
        let table_names: Vec<String> = stmt.query_map([], |row| row.get(0)).unwrap()
            .map(|res| res.unwrap())
            .collect();

        assert!(table_names.contains(&"notes".to_string()));
        assert!(table_names.contains(&"versions".to_string()));
    }
}
