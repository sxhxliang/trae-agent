//! # Git Utilities
//!
//! This module provides helper functions for interacting with Git repositories,
//! such as retrieving diffs and processing patch content.

use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;

/// Gets the git diff of the project.
///
/// If `base_commit` is `None` or an empty string, it performs a diff against the current
/// working tree (unstaged changes). If `base_commit` is specified, it diffs
/// between that commit and `HEAD`.
///
/// # Arguments
/// * `project_path`: Absolute path to the root of the git repository.
/// * `base_commit`: Optional commit hash, branch name, or tag to use as the base for the diff.
///
/// # Returns
/// A `Result` containing the diff output as a string, or an error if the `git diff` command fails
/// or its output is not valid UTF-8.
pub fn get_git_diff(project_path: &str, base_commit: Option<&str>) -> Result<String> {
    let mut cmd = Command::new("git");
    cmd.current_dir(Path::new(project_path));
    cmd.arg("--no-pager");
    cmd.arg("diff");

    if let Some(commit) = base_commit {
        if !commit.trim().is_empty() {
            // Ensure base_commit is not just whitespace
            cmd.arg(commit);
            cmd.arg("HEAD"); // Diff between base_commit and current HEAD
        }
    }
    // If base_commit is None or empty, it defaults to `git diff` (unstaged changes)

    let output = cmd
        .output()
        .with_context(|| format!("Failed to execute git diff in {}", project_path))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow::anyhow!(
            "git diff command failed with status {}: {}",
            output.status,
            stderr
        ));
    }

    String::from_utf8(output.stdout).with_context(|| "git diff output was not valid UTF-8")
}

/// Removes patches related to test files or directories from a given git diff string.
///
/// This function iterates through the lines of a diff. When it encounters a
/// `diff --git a/... b/...` line, it checks if the target file path (`b/...`)
/// matches common patterns for test files or directories (e.g., contains `/tests/`,
/// starts with `test_`, ends with `.spec.js`). If a test file chunk is identified,
/// subsequent lines belonging to that chunk are excluded from the output until
/// the next `diff --git` line is found.
///
/// This is useful for focusing on core code changes when `must_patch` is enabled.
/// The matching logic is based on conventions seen in Aider's SWE-bench tests.
///
/// # Arguments
/// * `model_patch`: A string containing the full git diff output.
///
/// # Returns
/// A string containing the filtered diff, with test file patches removed.
pub fn remove_patches_to_tests(model_patch: &str) -> String {
    let mut filtered_lines: Vec<String> = Vec::new();
    let mut is_tests_file_chunk = false;

    for line in model_patch.lines() {
        if line.starts_with("diff --git a/") {
            // Example: diff --git a/src/main.rs b/src/main.rs
            // We are interested in the 'to' file, which is the second path.
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                // Should be at least `diff --git a/path b/path`
                // let to_file_path = parts[3]; // e.g., b/src/main.rs
                let path_to_check = parts[3].strip_prefix("b/").unwrap_or(parts[3]); // Inlined to_file_path

                is_tests_file_chunk = path_to_check.contains("/test/") ||
                                      path_to_check.contains("/tests/") ||
                                      path_to_check.contains("/testing/") ||
                                      path_to_check.starts_with("test_") || // common prefix for test files
                                      path_to_check.ends_with("_test.py") || // common suffix
                                      path_to_check.ends_with("_tests.py") ||
                                      path_to_check.ends_with(".spec.js") || // common test file extensions
                                      path_to_check.ends_with(".test.js") ||
                                      path_to_check.ends_with(".spec.ts") ||
                                      path_to_check.ends_with(".test.ts") ||
                                      path_to_check.contains("test/") ||
                                      path_to_check.contains("tests/") ||
                                      path_to_check.contains("testing/") ||
                                      Path::new(path_to_check).file_name().is_some_and(|name| name.to_string_lossy().starts_with("test_")) || // Corrected
                                      path_to_check.ends_with("/tox.ini") ||
                                      path_to_check.ends_with("/pytest.ini");
            } else {
                // Malformed diff line, assume not a test file chunk for safety
                is_tests_file_chunk = false;
            }
        }

        if !is_tests_file_chunk {
            filtered_lines.push(line.to_string());
        }
    }
    filtered_lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    // Helper to init a git repo and make commits
    fn setup_git_repo(dir: &Path) -> Result<()> {
        Command::new("git").arg("init").current_dir(dir).status()?;
        Command::new("git")
            .args(["config", "user.name", "Test User"])
            .current_dir(dir)
            .status()?;
        Command::new("git")
            .args(["config", "user.email", "test@example.com"])
            .current_dir(dir)
            .status()?;
        Ok(())
    }

    fn commit_file(dir: &Path, filename: &str, content: &str) -> Result<()> {
        fs::write(dir.join(filename), content)?;
        Command::new("git")
            .arg("add")
            .arg(filename)
            .current_dir(dir)
            .status()?;
        Command::new("git")
            .arg("commit")
            .arg("-m")
            .arg(format!("add {}", filename))
            .current_dir(dir)
            .status()?;
        Ok(())
    }

    #[test]
    fn test_get_git_diff_no_changes() -> Result<()> {
        let dir = tempdir()?;
        setup_git_repo(dir.path())?;
        commit_file(dir.path(), "file.txt", "initial content")?;

        let diff = get_git_diff(dir.path().to_str().unwrap(), None)?;
        assert!(
            diff.is_empty(),
            "Diff should be empty when no changes are made after commit"
        );
        Ok(())
    }

    #[test]
    fn test_get_git_diff_with_unstaged_changes() -> Result<()> {
        let dir = tempdir()?;
        setup_git_repo(dir.path())?;
        commit_file(dir.path(), "file.txt", "initial content")?;
        fs::write(dir.path().join("file.txt"), "new content")?; // Unstaged change

        let diff = get_git_diff(dir.path().to_str().unwrap(), None)?;
        assert!(diff.contains("--- a/file.txt"));
        assert!(diff.contains("+++ b/file.txt"));
        assert!(diff.contains("-initial content"));
        assert!(diff.contains("+new content"));
        Ok(())
    }

    #[test]
    fn test_get_git_diff_between_commits() -> Result<()> {
        let dir = tempdir()?;
        setup_git_repo(dir.path())?;
        commit_file(dir.path(), "file.txt", "initial content")?;
        let initial_commit_hash = String::from_utf8(
            Command::new("git")
                .arg("rev-parse")
                .arg("HEAD")
                .current_dir(dir.path())
                .output()?
                .stdout,
        )?
        .trim()
        .to_string();

        commit_file(dir.path(), "file.txt", "new content")?; // This changes file.txt
        commit_file(dir.path(), "file2.txt", "another file")?; // This adds file2.txt

        // Diff between initial commit and current HEAD
        let diff = get_git_diff(dir.path().to_str().unwrap(), Some(&initial_commit_hash))?;

        assert!(diff.contains("--- a/file.txt")); // Change to file.txt
        assert!(diff.contains("+++ b/file.txt"));
        assert!(diff.contains("-initial content"));
        assert!(diff.contains("+new content"));

        assert!(diff.contains("diff --git a/file2.txt b/file2.txt")); // Addition of file2.txt
        assert!(diff.contains("new file mode"));
        assert!(diff.contains("+++ b/file2.txt"));
        assert!(diff.contains("+another file"));
        Ok(())
    }

    #[test]
    fn test_remove_patches_to_tests_simple() {
        let patch = r#"
diff --git a/src/main.py b/src/main.py
index 123..456 100644
--- a/src/main.py
+++ b/src/main.py
@@ -1,1 +1,1 @@
-print("hello")
+print("world")
diff --git a/tests/test_main.py b/tests/test_main.py
index abc..def 100644
--- a/tests/test_main.py
+++ b/tests/test_main.py
@@ -1,1 +1,1 @@
-assert True
+assert False
diff --git a/src/utils.py b/src/utils.py
index 789..012 100644
--- a/src/utils.py
+++ b/src/utils.py
@@ -1,1 +1,1 @@
 pass
+    # new line
"#
        .trim();
        let expected = r#"
diff --git a/src/main.py b/src/main.py
index 123..456 100644
--- a/src/main.py
+++ b/src/main.py
@@ -1,1 +1,1 @@
-print("hello")
+print("world")
diff --git a/src/utils.py b/src/utils.py
index 789..012 100644
--- a/src/utils.py
+++ b/src/utils.py
@@ -1,1 +1,1 @@
 pass
+    # new line
"#
        .trim();
        assert_eq!(remove_patches_to_tests(patch).trim(), expected);
    }

    #[test]
    fn test_remove_patches_to_tests_various_paths() {
        let patch_data = vec![
            ("diff --git a/myproject/test/file.py b/myproject/test/file.py\n- hello\n+ world", ""),
            ("diff --git a/myproject/tests/file.py b/myproject/tests/file.py\n- hello\n+ world", ""),
            ("diff --git a/myproject/testing/file.py b/myproject/testing/file.py\n- hello\n+ world", ""),
            ("diff --git a/myproject/test_file.py b/myproject/test_file.py\n- hello\n+ world", ""),
            ("diff --git a/myproject/file_test.py b/myproject/file_test.py\n- hello\n+ world", ""),
            ("diff --git a/myproject/tox.ini b/myproject/tox.ini\n- hello\n+ world", ""),
            ("diff --git a/src/app.py b/src/app.py\n- print(1)\n+ print(2)", "diff --git a/src/app.py b/src/app.py\n- print(1)\n+ print(2)"),
        ];

        for (patch, expected) in patch_data {
            assert_eq!(
                remove_patches_to_tests(patch).trim(),
                expected.trim(),
                "Failed for patch: {}",
                patch
            );
        }
    }
    #[test]
    fn test_remove_patches_to_tests_no_test_files() {
        let patch = r#"
diff --git a/src/main.rs b/src/main.rs
index e69de29..9586899 100644
--- a/src/main.rs
+++ b/src/main.rs
@@ -0,0 +1 @@
+fn main() { println!("Hello"); }
diff --git a/README.md b/README.md
index e69de29..197f867 100644
--- a/README.md
+++ b/README.md
@@ -0,0 +1 @@
+# My Project
"#
        .trim();
        assert_eq!(remove_patches_to_tests(patch).trim(), patch.trim());
    }

    #[test]
    fn test_remove_patches_to_tests_only_test_files() {
        let patch = r#"
diff --git a/tests/test_module.py b/tests/test_module.py
index e69de29..ba7d794 100644
--- a/tests/test_module.py
+++ b/tests/test_module.py
@@ -0,0 +1 @@
+def test_something(): pass
diff --git a/test_another.py b/test_another.py
index e69de29..067929c 100644
--- a/test_another.py
+++ b/test_another.py
@@ -0,0 +1 @@
+assert True
"#
        .trim();
        assert_eq!(remove_patches_to_tests(patch).trim(), "");
    }
}
