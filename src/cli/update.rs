use anyhow::{bail, Result};
use std::fs;
use std::process::Command;

const REPO: &str = "SeanoChang/ironvault";

pub fn run() -> Result<()> {
    let target = current_target();
    let binary_path = std::env::current_exe()?;

    // Get latest release tag from GitHub
    let tag_output = Command::new("gh")
        .args(["release", "view", "--repo", REPO, "--json", "tagName", "-q", ".tagName"])
        .output()?;

    if !tag_output.status.success() {
        bail!("failed to fetch latest release. Is `gh` installed and authenticated?");
    }

    let tag = String::from_utf8_lossy(&tag_output.stdout).trim().to_string();
    if tag.is_empty() {
        bail!("no releases found for {}", REPO);
    }

    let asset = format!("nark-{}-{}", tag, target);

    // Download to a temp file next to the binary
    let tmp_path = binary_path.with_extension("update");

    let download = Command::new("gh")
        .args([
            "release", "download", &tag,
            "--repo", REPO,
            "--pattern", &asset,
            "--output", &tmp_path.to_string_lossy(),
            "--clobber",
        ])
        .output()?;

    if !download.status.success() {
        let msg = String::from_utf8_lossy(&download.stderr);
        bail!("download failed: {}", msg.trim());
    }

    // Make executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&tmp_path, fs::Permissions::from_mode(0o755))?;
    }

    // Atomic replace
    fs::rename(&tmp_path, &binary_path)?;

    let out = serde_json::json!({
        "status": "ok",
        "version": tag,
        "target": target,
        "path": binary_path.display().to_string(),
    });
    println!("{}", serde_json::to_string_pretty(&out)?);

    Ok(())
}

fn current_target() -> &'static str {
    if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
        "aarch64-apple-darwin"
    } else if cfg!(target_os = "linux") && cfg!(target_arch = "x86_64") {
        "x86_64-unknown-linux-gnu"
    } else if cfg!(target_os = "macos") && cfg!(target_arch = "x86_64") {
        "x86_64-apple-darwin"
    } else if cfg!(target_os = "linux") && cfg!(target_arch = "aarch64") {
        "aarch64-unknown-linux-gnu"
    } else {
        "unknown"
    }
}
