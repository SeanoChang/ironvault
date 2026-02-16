use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Frontmatter {
    title: String,
    author: String,
    domain: String,
    intent: String,
    kind: String,
    trust: String,
    status: String,
    tags: Vec<String>,
}

