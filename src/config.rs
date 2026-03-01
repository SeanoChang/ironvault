use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub search: SearchConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    pub weights: BlendWeights,
    pub recency: RecencyConfig,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct BlendWeights {
    pub cosine: f64,
    pub graph: f64,
    pub importance: f64,
    pub recency: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RecencyConfig {
    pub lambda: f64,
}

impl Default for BlendWeights {
    fn default() -> Self {
        Self {
            cosine: 0.50,
            graph: 0.25,
            importance: 0.15,
            recency: 0.10,
        }
    }
}

impl Default for RecencyConfig {
    fn default() -> Self {
        Self {
            lambda: 0.05,
        }
    }
}

pub fn load(vault_dir: &Path) -> Result<Config> {
    let path = vault_dir.join("config.toml");
    if path.exists() {
        let content = std::fs::read_to_string(&path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    } else {
        Ok(Config::default())
    }
}
