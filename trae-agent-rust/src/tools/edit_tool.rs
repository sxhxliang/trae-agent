use super::base::{Tool, ToolError, ToolExecResult, ToolParameter};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, instrument};

const SNIPPET_LINES: usize = 4;
const TAB_WIDTH: usize = 8; // Define tab width

#[derive(Deserialize, Debug)]
struct EditToolArgs {
    command: String,
    path: String,
    file_text: Option<String>,
    insert_line: Option<i64>,
    new_str: Option<String>,
    old_str: Option<String>,
    view_range: Option<Vec<i64>>,
}

pub struct EditTool;

impl EditTool {
    pub fn new() -> Self {
        EditTool
    }

    fn expand_tabs(text: &str) -> String {
        text.replace('\t', &" ".repeat(TAB_WIDTH))
    }

    fn validate_path_exists(&self, p: &Path, command_name: &str) -> Result<(), ToolError> {
        if !p.exists() && command_name != "create" {
            return Err(ToolError::NotFound(format!(
                "Path {} does not exist for command '{}'.",
                p.display(),
                command_name
            )));
        }
        if p.exists() && command_name == "create" {
            return Err(ToolError::ExecutionFailed(format!(
                "File already exists at: {}. Cannot overwrite files using command `create`.",
                p.display()
            )));
        }
        Ok(())
    }

    fn validate_path_is_file(&self, p: &Path) -> Result<(), ToolError> {
        if !p.is_file() {
            return Err(ToolError::InvalidArguments {
                tool_name: self.get_name(),
                message: format!("Path {} is not a file.", p.display()),
            });
        }
        Ok(())
    }

    fn validate_path_is_dir(&self, p: &Path) -> Result<(), ToolError> {
        if !p.is_dir() {
            return Err(ToolError::InvalidArguments {
                tool_name: self.get_name(),
                message: format!("Path {} is not a directory.", p.display()),
            });
        }
        Ok(())
    }

    async fn view_file(
        &self,
        path: &Path,
        view_range: Option<&Vec<i64>>,
    ) -> Result<ToolExecResult, ToolError> {
        self.validate_path_is_file(path)?;
        let raw_content = fs::read_to_string(path).await.map_err(|e| {
            ToolError::ExecutionFailed(format!("Failed to read file {}: {}", path.display(), e))
        })?;
        let content_expanded = Self::expand_tabs(&raw_content);

        let mut display_start_line_1_indexed: usize = 1;
        let mut content_to_display = content_expanded;

        if let Some(range) = view_range {
            if range.len() != 2 {
                return Err(ToolError::InvalidArguments {
                    tool_name: self.get_name(),
                    message: "view_range must contain exactly two integers [start, end]."
                        .to_string(),
                });
            }
            let start_line_req = range[0];
            let end_line_req = range[1];

            let lines: Vec<&str> = content_to_display.lines().collect();
            let n_lines_file = lines.len();

            if start_line_req < 1 || start_line_req as usize > n_lines_file {
                return Err(ToolError::InvalidArguments {
                    tool_name: self.get_name(),
                    message: format!(
                        "Invalid start_line {} for file with {} lines.",
                        start_line_req, n_lines_file
                    ),
                });
            }
            display_start_line_1_indexed = start_line_req as usize;
            let start_idx_0_indexed = display_start_line_1_indexed - 1;

            if end_line_req == -1 {
                content_to_display = lines
                    .iter()
                    .skip(start_idx_0_indexed)
                    .copied()
                    .collect::<Vec<&str>>()
                    .join("\n");
            } else {
                if end_line_req < start_line_req || end_line_req as usize > n_lines_file {
                    return Err(ToolError::InvalidArguments {
                        tool_name: self.get_name(),
                        message: format!(
                            "Invalid end_line {} for file with {} lines and start_line {}.",
                            end_line_req, n_lines_file, start_line_req
                        ),
                    });
                }
                let end_idx_0_indexed = end_line_req as usize;
                content_to_display = lines
                    .iter()
                    .skip(start_idx_0_indexed)
                    .take(end_idx_0_indexed - start_idx_0_indexed)
                    .copied()
                    .collect::<Vec<&str>>()
                    .join("\n");
            }
        }

        let numbered_content = content_to_display
            .lines()
            .enumerate()
            .map(|(i, line)| format!("{:6}\t{}", i + display_start_line_1_indexed, line))
            .collect::<Vec<String>>()
            .join("\n");

        Ok(ToolExecResult {
            output: Some(format!(
                "Here's the content of {}:\n{}",
                path.display(),
                numbered_content
            )),
            error: None,
            error_code: 0,
        })
    }

    async fn view_dir(&self, path: &Path) -> Result<ToolExecResult, ToolError> {
        self.validate_path_is_dir(path)?;
        let mut entries = Vec::new();

        let mut l1_read_dir = fs::read_dir(path).await.map_err(|e| {
            ToolError::ExecutionFailed(format!(
                "Failed to read directory {}: {}",
                path.display(),
                e
            ))
        })?;
        while let Some(entry) = l1_read_dir.next_entry().await.map_err(|e| {
            ToolError::ExecutionFailed(format!(
                "Error iterating directory {}: {}",
                path.display(),
                e
            ))
        })? {
            let entry_path = entry.path();
            let name = entry_path.file_name().unwrap_or_default().to_string_lossy();
            if name.starts_with('.') {
                continue;
            }

            if entry_path.is_dir() {
                entries.push(format!("{}/ (dir)", name));
                let mut l2_read_dir = fs::read_dir(&entry_path).await.map_err(|e| {
                    ToolError::ExecutionFailed(format!(
                        "Failed to read subdirectory {}: {}",
                        entry_path.display(),
                        e
                    ))
                })?;
                while let Some(sub_entry) = l2_read_dir.next_entry().await.map_err(|e| {
                    ToolError::ExecutionFailed(format!(
                        "Error iterating subdirectory {}: {}",
                        entry_path.display(),
                        e
                    ))
                })? {
                    let sub_entry_path = sub_entry.path();
                    let sub_name = sub_entry_path
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy();
                    if sub_name.starts_with('.') {
                        continue;
                    }
                    if sub_entry_path.is_dir() {
                        entries.push(format!("  {}/ (dir)", sub_name));
                    } else if sub_entry_path.is_file() {
                        entries.push(format!("  {} (file)", sub_name));
                    }
                }
            } else if entry_path.is_file() {
                entries.push(format!("{} (file)", name));
            }
        }
        Ok(ToolExecResult {
            output: Some(format!(
                "Contents of directory {}:\n{}",
                path.display(),
                entries.join("\n")
            )),
            error: None,
            error_code: 0,
        })
    }

    fn make_snippet_output(
        &self,
        content: &str,
        file_descriptor: &str,
        start_line_num: usize,
    ) -> String {
        let numbered_content = content
            .lines()
            .enumerate()
            .map(|(i, line)| format!("{:6}\t{}", i + start_line_num, line))
            .collect::<Vec<String>>()
            .join("\n");
        format!("Here's {}:\n{}\n", file_descriptor, numbered_content)
    }
}

#[async_trait]
impl Tool for EditTool {
    fn get_name(&self) -> String {
        "str_replace_based_edit_tool".to_string()
    }

    fn get_description(&self) -> String {
        "Tool for viewing, creating, and editing files. \
        Supports viewing file/directory content (up to 2 levels for dirs), creating new files, \
        replacing exact string occurrences in files (tabs expanded to 8 spaces for matching), \
        and inserting text at specific lines (tabs in input string also expanded). \
        File content with tabs will be converted to spaces upon edit. \
        Be careful with paths (must be absolute) and ensure strings for replacement are unique after tab expansion."
            .to_string()
    }

    fn get_parameters(&self) -> Vec<ToolParameter> {
        vec![
            ToolParameter {
                name: "command".to_string(), param_type: "string".to_string(),
                description: "The command to run: view, create, str_replace, insert.".to_string(),
                is_required: true, enum_values: Some(vec!["view".into(), "create".into(), "str_replace".into(), "insert".into()]),
                items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "path".to_string(), param_type: "string".to_string(),
                description: "Absolute path to the file or directory.".to_string(),
                is_required: true, enum_values: None, items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "file_text".to_string(), param_type: "string".to_string(),
                description: "Content for the 'create' command. Tabs will be preserved as-is during creation.".to_string(),
                is_required: false, enum_values: None, items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "insert_line".to_string(), param_type: "integer".to_string(),
                description: "Line number (1-indexed) AFTER which to insert for 'insert' command. Use 0 to insert at the beginning.".to_string(),
                is_required: false, enum_values: None, items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "new_str".to_string(), param_type: "string".to_string(),
                description: "Text to insert for 'insert', or the replacement text for 'str_replace'. Tabs will be expanded to 8 spaces.".to_string(),
                is_required: false, enum_values: None, items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "old_str".to_string(), param_type: "string".to_string(),
                description: "Exact text to replace for 'str_replace' command. Tabs will be expanded to 8 spaces for matching.".to_string(),
                is_required: false, enum_values: None, items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "view_range".to_string(), param_type: "array".to_string(),
                description: "Optional [start, end] line numbers (1-indexed) for 'view' command. Use -1 for end to view till end of file.".to_string(),
                is_required: false,
                items: Some(Box::new(ToolParameter {
                    name: "line_number".to_string(),
                    param_type: "integer".to_string(),
                    description: "A line number".to_string(),
                    is_required: true,
                    enum_values: None,
                    items: None, properties: None, required: vec![]
                })),
                properties: None, required: vec![],
                enum_values: None,
            },
        ]
    }

    #[instrument(skip(self, arguments), fields(tool_name = %self.get_name()))]
    async fn execute(&self, arguments: Value) -> Result<ToolExecResult, ToolError> {
        debug!(args = ?arguments, "Executing edit tool");
        let args: EditToolArgs =
            serde_json::from_value(arguments.clone()).map_err(|e| ToolError::InvalidArguments {
                tool_name: self.get_name(),
                message: format!("Failed to parse arguments: {}. Args: {:?}", e, arguments),
            })?;

        let path_buf = PathBuf::from(&args.path);
        if !path_buf.is_absolute() {
            return Err(ToolError::InvalidArguments {
                tool_name: self.get_name(),
                message: format!("Path '{}' must be absolute.", args.path),
            });
        }

        self.validate_path_exists(&path_buf, &args.command)?;

        match args.command.as_str() {
            "view" => {
                if path_buf.is_dir() {
                    if args.view_range.is_some() {
                        return Err(ToolError::InvalidArguments {
                            tool_name: self.get_name(),
                            message: "view_range is not allowed for directory view.".to_string(),
                        });
                    }
                    self.view_dir(&path_buf).await
                } else {
                    self.view_file(&path_buf, args.view_range.as_ref()).await
                }
            }
            "create" => {
                let content = args.file_text.ok_or_else(|| ToolError::InvalidArguments {
                    tool_name: self.get_name(),
                    message: "'file_text' is required for 'create' command.".to_string(),
                })?;
                fs::write(&path_buf, &content).await.map_err(|e| {
                    ToolError::ExecutionFailed(format!(
                        "Failed to create file {}: {}",
                        path_buf.display(),
                        e
                    ))
                })?;
                Ok(ToolExecResult {
                    output: Some(format!(
                        "File created successfully at: {}",
                        path_buf.display()
                    )),
                    error: None,
                    error_code: 0,
                })
            }
            "str_replace" => {
                self.validate_path_is_file(&path_buf)?;
                let old_s_raw = args.old_str.ok_or_else(|| ToolError::InvalidArguments {
                    tool_name: self.get_name(),
                    message: "'old_str' is required for 'str_replace'".to_string(),
                })?;
                let new_s_raw = args.new_str.unwrap_or_default();

                let old_s_expanded = Self::expand_tabs(&old_s_raw);
                let new_s_expanded = Self::expand_tabs(&new_s_raw);

                let file_content_raw = fs::read_to_string(&path_buf).await.map_err(|e| {
                    ToolError::ExecutionFailed(format!(
                        "Failed to read file {}: {}",
                        path_buf.display(),
                        e
                    ))
                })?;
                let content_expanded = Self::expand_tabs(&file_content_raw);

                let occurrences = content_expanded.matches(&old_s_expanded).count();
                if occurrences == 0 {
                    return Err(ToolError::ExecutionFailed(format!(
                        "Pattern '{}' (with tabs expanded to {} spaces) not found in file {}.",
                        old_s_raw,
                        TAB_WIDTH,
                        path_buf.display()
                    )));
                }
                if occurrences > 1 {
                    return Err(ToolError::ExecutionFailed(format!("Pattern '{}' (with tabs expanded to {} spaces) found {} times in file {}. Replacement must be unique.", old_s_raw, TAB_WIDTH, occurrences, path_buf.display())));
                }

                let replacement_line_idx = content_expanded
                    .lines()
                    .position(|line| line.contains(&old_s_expanded))
                    .unwrap_or(0);
                let new_content_expanded =
                    content_expanded.replace(&old_s_expanded, &new_s_expanded);

                fs::write(&path_buf, &new_content_expanded)
                    .await
                    .map_err(|e| {
                        ToolError::ExecutionFailed(format!(
                            "Failed to write to file {}: {}",
                            path_buf.display(),
                            e
                        ))
                    })?;

                let snippet_start_line_idx = replacement_line_idx.saturating_sub(SNIPPET_LINES);
                let num_new_lines = new_s_expanded.lines().count();
                let num_content_lines = new_content_expanded.lines().count();

                let snippet_end_line_idx_exclusive =
                    (replacement_line_idx + num_new_lines + SNIPPET_LINES).min(num_content_lines);

                let snippet: String = new_content_expanded
                    .lines()
                    .skip(snippet_start_line_idx)
                    .take(snippet_end_line_idx_exclusive.saturating_sub(snippet_start_line_idx))
                    .collect::<Vec<&str>>()
                    .join("\n");

                let output_msg = format!(
                    "File {} edited successfully. {}",
                    path_buf.display(),
                    self.make_snippet_output(
                        &snippet,
                        &format!("a snippet of {}", path_buf.display()),
                        snippet_start_line_idx + 1
                    )
                );
                Ok(ToolExecResult {
                    output: Some(output_msg),
                    error: None,
                    error_code: 0,
                })
            }
            "insert" => {
                self.validate_path_is_file(&path_buf)?;
                let line_num_1_indexed =
                    args.insert_line
                        .ok_or_else(|| ToolError::InvalidArguments {
                            tool_name: self.get_name(),
                            message:
                                "'insert_line' (1-indexed, 0 for start) is required for 'insert'"
                                    .to_string(),
                        })?;
                let text_to_insert_raw =
                    args.new_str.ok_or_else(|| ToolError::InvalidArguments {
                        tool_name: self.get_name(),
                        message: "'new_str' is required for 'insert'".to_string(),
                    })?;

                if line_num_1_indexed < 0 {
                    return Err(ToolError::InvalidArguments{tool_name: self.get_name(), message: "insert_line must be non-negative (0 for beginning, N for after Nth line).".to_string()});
                }

                let text_to_insert_expanded = Self::expand_tabs(&text_to_insert_raw);
                let new_lines_to_insert_expanded: Vec<String> =
                    text_to_insert_expanded.lines().map(String::from).collect();

                let file_content_raw = fs::read_to_string(&path_buf).await.map_err(|e| {
                    ToolError::ExecutionFailed(format!(
                        "Failed to read file {}: {}",
                        path_buf.display(),
                        e
                    ))
                })?;
                let mut lines_expanded: Vec<String> = Self::expand_tabs(&file_content_raw)
                    .lines()
                    .map(String::from)
                    .collect();

                let true_insert_idx = if line_num_1_indexed == 0 {
                    0
                } else {
                    line_num_1_indexed as usize
                };

                if true_insert_idx > lines_expanded.len() {
                    return Err(ToolError::InvalidArguments{tool_name: self.get_name(), message: format!("insert_line {} is out of bounds for file with {} lines. Max valid is {} (to append).", line_num_1_indexed, lines_expanded.len(), lines_expanded.len())});
                }

                for (i, nl) in new_lines_to_insert_expanded.iter().enumerate() {
                    lines_expanded.insert(true_insert_idx + i, nl.clone());
                }

                let new_content_expanded = lines_expanded.join("\n");
                fs::write(&path_buf, &new_content_expanded)
                    .await
                    .map_err(|e| {
                        ToolError::ExecutionFailed(format!(
                            "Failed to write to file {}: {}",
                            path_buf.display(),
                            e
                        ))
                    })?;

                let snippet_actual_start_idx = true_insert_idx.saturating_sub(SNIPPET_LINES);
                let snippet_actual_end_idx_exclusive =
                    (true_insert_idx + new_lines_to_insert_expanded.len() + SNIPPET_LINES)
                        .min(lines_expanded.len());

                let snippet_display_start_line_num = snippet_actual_start_idx + 1;

                let snippet_str = if snippet_actual_start_idx < snippet_actual_end_idx_exclusive {
                    lines_expanded[snippet_actual_start_idx..snippet_actual_end_idx_exclusive]
                        .join("\n")
                } else {
                    String::new()
                };

                let output_msg = format!(
                    "Text inserted into {}. {}",
                    path_buf.display(),
                    self.make_snippet_output(
                        &snippet_str,
                        &format!("a snippet of {}", path_buf.display()),
                        snippet_display_start_line_num
                    )
                );
                Ok(ToolExecResult {
                    output: Some(output_msg),
                    error: None,
                    error_code: 0,
                })
            }
            _ => Err(ToolError::InvalidArguments {
                tool_name: self.get_name(),
                message: format!("Unknown command: {}", args.command),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokio::runtime::Runtime;

    fn run_async_test<F, Fut>(test_fn: F)
    where
        F: FnOnce(EditTool, PathBuf) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            let tool = EditTool::new();
            let temp_dir = tempdir().unwrap();
            let base_path = temp_dir.path().to_path_buf();
            test_fn(tool, base_path).await;
        });
    }

    #[test]
    fn test_create_file_success() {
        run_async_test(|tool, base_path| async move {
            let file_path = base_path.join("test_create.txt");
            let args = serde_json::json!({
                "command": "create",
                "path": file_path.to_str().unwrap(),
                "file_text": "Hello, world!"
            });
            let result = tool.execute(args).await.unwrap();
            assert!(
                result.error.is_none(),
                "Create file should succeed. Error: {:?}",
                result.error
            );
            assert_eq!(result.error_code, 0);
            assert!(result.output.unwrap().contains("File created successfully"));
            assert_eq!(
                fs::read_to_string(&file_path).await.unwrap(),
                "Hello, world!"
            );
        });
    }

    #[test]
    fn test_create_file_already_exists() {
        run_async_test(|tool, base_path| async move {
            let file_path = base_path.join("test_exists.txt");
            fs::write(&file_path, "initial").await.unwrap();

            let args = serde_json::json!({
                "command": "create",
                "path": file_path.to_str().unwrap(),
                "file_text": "new text"
            });
            let result = tool.execute(args).await;
            assert!(result.is_err());
            if let Err(ToolError::ExecutionFailed(msg)) = result {
                assert!(msg.contains("File already exists"));
            } else {
                panic!(
                    "Expected ExecutionFailed error for existing file, got {:?}",
                    result
                );
            }
        });
    }

    #[test]
    fn test_view_file_simple() {
        run_async_test(|tool, base_path| async move {
            let file_path = base_path.join("test_view.txt");
            let content = "Line 1\nLine 2\nLine 3";
            fs::write(&file_path, content).await.unwrap();

            let args = serde_json::json!({
                "command": "view",
                "path": file_path.to_str().unwrap()
            });
            let result = tool.execute(args).await.unwrap();
            assert!(
                result.error.is_none(),
                "View file should succeed. Error: {:?}",
                result.error
            );
            assert_eq!(result.error_code, 0);
            let output = result.output.unwrap();
            assert!(output.contains("1\tLine 1"));
            assert!(output.contains("2\tLine 2"));
            assert!(output.contains("3\tLine 3"));
        });
    }

    #[test]
    fn test_view_file_with_range() {
        run_async_test(|tool, base_path| async move {
            let file_path = base_path.join("test_view_range.txt");
            let content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5";
            fs::write(&file_path, content).await.unwrap();

            let args = serde_json::json!({
                "command": "view",
                "path": file_path.to_str().unwrap(),
                "view_range": [2, 4]
            });
            let result = tool.execute(args).await.unwrap();
            assert_eq!(
                result.error_code, 0,
                "Expected success. Output: {:?}, Error: {:?}",
                result.output, result.error
            );
            let output = result.output.unwrap();
            assert!(output.contains("2\tLine 2"));
            assert!(output.contains("3\tLine 3"));
            assert!(output.contains("4\tLine 4"));
            assert!(!output.contains("1\tLine 1"));
            assert!(!output.contains("5\tLine 5"));

            let args_to_end = serde_json::json!({
                "command": "view",
                "path": file_path.to_str().unwrap(),
                "view_range": [3, -1]
            });
            let result_to_end = tool.execute(args_to_end).await.unwrap();
            assert_eq!(
                result_to_end.error_code, 0,
                "Expected success. Output: {:?}, Error: {:?}",
                result_to_end.output, result_to_end.error
            );
            let output_to_end = result_to_end.output.unwrap();
            assert!(output_to_end.contains("3\tLine 3"));
            assert!(output_to_end.contains("4\tLine 4"));
            assert!(output_to_end.contains("5\tLine 5"));
            assert!(!output_to_end.contains("1\tLine 1"));
            assert!(!output_to_end.contains("2\tLine 2"));
        });
    }

    #[test]
    fn test_view_file_invalid_ranges() {
        run_async_test(|tool, base_path| async move {
            let file_path = base_path.join("test_view_invalid_range.txt");
            let content = "Line 1\nLine 2\nLine 3";
            fs::write(&file_path, content).await.unwrap();

            let args_start_too_low = serde_json::json!({"command": "view", "path": file_path.to_str().unwrap(), "view_range": [0, 2]});
            assert!(tool.execute(args_start_too_low).await.is_err());

            let args_start_too_high = serde_json::json!({"command": "view", "path": file_path.to_str().unwrap(), "view_range": [4, 4]});
            assert!(tool.execute(args_start_too_high).await.is_err());

            let args_end_before_start = serde_json::json!({"command": "view", "path": file_path.to_str().unwrap(), "view_range": [3, 2]});
            assert!(tool.execute(args_end_before_start).await.is_err());

            let args_end_too_high = serde_json::json!({"command": "view", "path": file_path.to_str().unwrap(), "view_range": [1, 5]});
            assert!(tool.execute(args_end_too_high).await.is_err());
        });
    }

    #[test]
    fn test_view_file_with_tabs() {
        run_async_test(|tool, base_path| async move {
            let file_path = base_path.join("test_view_tabs.txt");
            let content = "Hello\tWorld\n\tIndented";
            fs::write(&file_path, content).await.unwrap();

            let args = serde_json::json!({
                "command": "view",
                "path": file_path.to_str().unwrap()
            });
            let result = tool.execute(args).await.unwrap();
            assert_eq!(
                result.error_code, 0,
                "Expected success. Output: {:?}, Error: {:?}",
                result.output, result.error
            );
            let output = result.output.unwrap();
            let expected_line1 = format!("1\tHello{}World", " ".repeat(TAB_WIDTH));
            let expected_line2 = format!("2\t{}Indented", " ".repeat(TAB_WIDTH));
            assert!(output.contains(&expected_line1), "Output was: {}", output);
            assert!(output.contains(&expected_line2), "Output was: {}", output);
        });
    }

    #[test]
    fn test_view_dir() {
        run_async_test(|tool, base_path| async move {
            let dir_path = base_path.join("test_dir_view");
            fs::create_dir_all(dir_path.join("subdir1/subsubdir"))
                .await
                .unwrap();
            fs::write(dir_path.join("file1.txt"), "content1")
                .await
                .unwrap();
            fs::write(dir_path.join("subdir1/subfile1.txt"), "sub_content1")
                .await
                .unwrap();
            fs::write(dir_path.join(".hiddenfile"), "hidden")
                .await
                .unwrap();
            fs::create_dir(dir_path.join(".hiddendir")).await.unwrap();
            fs::write(dir_path.join("subdir1/.hidden_sub_file.txt"), "hidden_sub")
                .await
                .unwrap();

            let args = serde_json::json!({
                "command": "view",
                "path": dir_path.to_str().unwrap()
            });
            let result = tool.execute(args).await.unwrap();
            assert_eq!(
                result.error_code, 0,
                "Expected success. Output: {:?}, Error: {:?}",
                result.output, result.error
            );
            let output = result.output.unwrap();

            assert!(output.contains("file1.txt (file)"));
            assert!(output.contains("subdir1/ (dir)"));
            assert!(output.contains("  subfile1.txt (file)"));
            assert!(output.contains("  subsubdir/ (dir)"));
            assert!(!output.contains(".hiddenfile"));
            assert!(!output.contains(".hiddendir/"));
            assert!(!output.contains(".hidden_sub_file.txt"));
        });
    }

    #[test]
    fn test_str_replace_simple_success() {
        run_async_test(|tool, base_path| async move {
            let file_path = base_path.join("test_replace.txt");
            let initial_content = "Hello world, this is a test.\nAnother line with world.";
            fs::write(&file_path, initial_content).await.unwrap();

            let unique_old_str = "this is a test";
            let args_unique = serde_json::json!({
                "command": "str_replace",
                "path": file_path.to_str().unwrap(),
                "old_str": unique_old_str,
                "new_str": "Rust is amazing"
            });

            let result = tool.execute(args_unique).await.unwrap();
            assert_eq!(
                result.error_code, 0,
                "Expected success. Output: {:?}, Error: {:?}",
                result.output, result.error
            );
            let new_content_expanded = fs::read_to_string(&file_path).await.unwrap();
            assert_eq!(
                new_content_expanded,
                "Hello world, Rust is amazing.\nAnother line with world."
            );
            assert!(result.output.unwrap().contains("edited successfully"));
        });
    }

    #[test]
    fn test_str_replace_not_found() {
        run_async_test(|tool, base_path| async move {
            let file_path = base_path.join("test_replace_notfound.txt");
            fs::write(&file_path, "Hello world").await.unwrap();
            let args = serde_json::json!({
                "command": "str_replace",
                "path": file_path.to_str().unwrap(),
                "old_str": "Rust",
                "new_str": "Monde"
            });
            let result = tool.execute(args).await;
            assert!(result.is_err());
            match result.err().unwrap() {
                ToolError::ExecutionFailed(msg) => assert!(msg.contains("not found in file")),
                _ => panic!("Expected ExecutionFailed for pattern not found."),
            }
        });
    }

    #[test]
    fn test_str_replace_multiple_found_error() {
        run_async_test(|tool, base_path| async move {
            let file_path = base_path.join("test_replace_multiple.txt");
            fs::write(&file_path, "world world world").await.unwrap();
            let args = serde_json::json!({
                "command": "str_replace",
                "path": file_path.to_str().unwrap(),
                "old_str": "world",
                "new_str": "Rust"
            });
            let result = tool.execute(args).await;
            assert!(result.is_err());
            match result.err().unwrap() {
                ToolError::ExecutionFailed(msg) => assert!(msg.contains("found 3 times")),
                _ => panic!("Expected ExecutionFailed for multiple occurrences."),
            }
        });
    }

    #[test]
    fn test_str_replace_with_tabs() {
        run_async_test(|tool, base_path| async move {
            let file_path = base_path.join("test_replace_tabs.txt");
            fs::write(&file_path, "hello\t\tworld").await.unwrap();

            let args = serde_json::json!({
                "command": "str_replace",
                "path": file_path.to_str().unwrap(),
                "old_str": "hello\t\tworld",
                "new_str": "bye\tRust"
            });
            let result = tool.execute(args).await.unwrap();
            assert_eq!(
                result.error_code, 0,
                "Expected success. Output: {:?}, Error: {:?}",
                result.output, result.error
            );
            let new_content_on_disk = fs::read_to_string(&file_path).await.unwrap();

            let expected_disk_content = format!("bye{}Rust", " ".repeat(TAB_WIDTH));
            assert_eq!(new_content_on_disk, expected_disk_content);
        });
    }

    #[test]
    fn test_insert_beginning() {
        run_async_test(|tool, base_path| async move {
            let file_path = base_path.join("test_insert_begin.txt");
            fs::write(&file_path, "Line 1\nLine 2").await.unwrap();
            let args = serde_json::json!({
                "command": "insert",
                "path": file_path.to_str().unwrap(),
                "insert_line": 0,
                "new_str": "NewLine 0"
            });
            let result = tool.execute(args).await.unwrap();
            assert_eq!(
                result.error_code, 0,
                "Expected success. Output: {:?}, Error: {:?}",
                result.output, result.error
            );
            let new_content = fs::read_to_string(&file_path).await.unwrap();
            assert_eq!(new_content, "NewLine 0\nLine 1\nLine 2");
        });
    }

    #[test]
    fn test_insert_middle() {
        run_async_test(|tool, base_path| async move {
            let file_path = base_path.join("test_insert_middle.txt");
            fs::write(&file_path, "Line 1\nLine 3").await.unwrap();
            let args = serde_json::json!({
                "command": "insert",
                "path": file_path.to_str().unwrap(),
                "insert_line": 1,
                "new_str": "Line 2"
            });
            let result = tool.execute(args).await.unwrap();
            assert_eq!(
                result.error_code, 0,
                "Expected success. Output: {:?}, Error: {:?}",
                result.output, result.error
            );
            let new_content = fs::read_to_string(&file_path).await.unwrap();
            assert_eq!(new_content, "Line 1\nLine 2\nLine 3");
        });
    }

    #[test]
    fn test_insert_end() {
        run_async_test(|tool, base_path| async move {
            let file_path = base_path.join("test_insert_end.txt");
            fs::write(&file_path, "Line 1\nLine 2").await.unwrap();
            let args = serde_json::json!({
                "command": "insert",
                "path": file_path.to_str().unwrap(),
                "insert_line": 2,
                "new_str": "Line 3"
            });
            let result = tool.execute(args).await.unwrap();
            assert_eq!(
                result.error_code, 0,
                "Expected success. Output: {:?}, Error: {:?}",
                result.output, result.error
            );
            let new_content = fs::read_to_string(&file_path).await.unwrap();
            assert_eq!(new_content, "Line 1\nLine 2\nLine 3");
        });
    }

    #[test]
    fn test_insert_with_tabs() {
        run_async_test(|tool, base_path| async move {
            let file_path = base_path.join("test_insert_tabs.txt");
            fs::write(&file_path, "First\tLine").await.unwrap();
            let args = serde_json::json!({
                "command": "insert",
                "path": file_path.to_str().unwrap(),
                "insert_line": 1,
                "new_str": "\tNew\tLine"
            });
            let result = tool.execute(args).await.unwrap();
            assert_eq!(
                result.error_code, 0,
                "Expected success. Output: {:?}, Error: {:?}",
                result.output, result.error
            );
            let new_content = fs::read_to_string(&file_path).await.unwrap();
            let line1_expanded = format!("First{}Line", " ".repeat(TAB_WIDTH));
            let line2_expanded =
                format!("{}New{}Line", " ".repeat(TAB_WIDTH), " ".repeat(TAB_WIDTH));
            let expected_final_content = format!("{}\n{}", line1_expanded, line2_expanded);
            assert_eq!(new_content, expected_final_content);
        });
    }

    #[test]
    fn test_path_not_absolute_error() {
        run_async_test(|tool, _base_path| async move {
            let args = serde_json::json!({
                "command": "view",
                "path": "relative/path.txt"
            });
            let result = tool.execute(args).await;
            assert!(result.is_err());
            match result.err().unwrap() {
                ToolError::InvalidArguments { message, .. } => {
                    assert!(message.contains("must be absolute"))
                }
                _ => panic!("Expected InvalidArguments for relative path."),
            }
        });
    }

    #[test]
    fn test_path_not_exists_error() {
        run_async_test(|tool, base_path| async move {
            let non_existent_path = base_path.join("non_existent.txt");
            let args = serde_json::json!({
                "command": "view",
                "path": non_existent_path.to_str().unwrap()
            });
            let result = tool.execute(args).await;
            assert!(result.is_err());
            match result.err().unwrap() {
                ToolError::NotFound(msg) => {
                    assert!(msg.contains("does not exist for command 'view'"))
                }
                _ => panic!("Expected NotFound error."),
            }
        });
    }
}
