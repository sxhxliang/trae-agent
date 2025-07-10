use super::base::{Tool, ToolError, ToolExecResult, ToolParameter};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use std::process::Stdio;
use tokio::process::Command;
use tracing::{debug, error, instrument};

const MAX_BASH_OUTPUT_LEN: usize = 16000; // Max length for stdout/stderr before truncation
const TRUNCATED_BASH_MESSAGE: &str = "<response clipped><NOTE>To save on context only part of this output has been shown. You might want to use file redirection or more specific commands to manage large outputs.</NOTE>";

#[derive(Deserialize, Debug)]
struct BashToolArgs {
    command: String,
    timeout: Option<u64>, // Timeout in seconds
    working_directory: Option<String>,
}

pub struct BashTool;

impl BashTool {
    pub fn new() -> Self {
        BashTool
    }

    fn maybe_truncate(content: String) -> String {
        if content.len() > MAX_BASH_OUTPUT_LEN {
            // Ensure TRUNCATED_BASH_MESSAGE itself isn't longer than MAX_BASH_OUTPUT_LEN
            if TRUNCATED_BASH_MESSAGE.len() >= MAX_BASH_OUTPUT_LEN {
                // If the message is too long, just truncate content to max length
                // and don't append the truncation message. Or return only the message.
                // For now, let's prioritize showing the start of the content.
                let mut truncated = content;
                truncated.truncate(MAX_BASH_OUTPUT_LEN);
                return truncated; // Or consider just returning TRUNCATED_BASH_MESSAGE if it fits
            }

            let mut truncated = content;
            // Calculate length to keep from original content
            let content_len_to_keep = MAX_BASH_OUTPUT_LEN.saturating_sub(TRUNCATED_BASH_MESSAGE.len());
            truncated.truncate(content_len_to_keep);
            truncated.push_str(TRUNCATED_BASH_MESSAGE);
            truncated
        } else {
            content
        }
    }
}

#[async_trait] // Added
impl Tool for BashTool {
    fn get_name(&self) -> String {
        "bash".to_string()
    }

    fn get_description(&self) -> String {
        "Executes a shell command and returns its stdout and stderr. \
        Use this for running scripts, system commands, etc. \
        Ensure commands are safe and necessary. \
        The command is executed in a temporary shell (sh -c 'command')."
            .to_string()
    }

    fn get_parameters(&self) -> Vec<ToolParameter> {
        // This definition of ToolParameter is the complex, recursive one.
        // For BashTool, the parameters are simple.
        // We need to ensure this maps correctly to the llm_types::FunctionParameterProperty
        // in the default get_json_definition or override get_json_definition.
        // For now, assuming simple parameters will be converted appropriately by the default.
        vec![
            ToolParameter {
                name: "command".to_string(),
                param_type: "string".to_string(),
                description: "The shell command to execute.".to_string(),
                is_required: true, // Mark as required
                enum_values: None,
                items: None,
                properties: None,
                required: vec![],
            },
            ToolParameter {
                name: "timeout".to_string(),
                param_type: "integer".to_string(),
                description: "Optional timeout in seconds for the command execution.".to_string(),
                is_required: false,
                enum_values: None,
                items: None,
                properties: None,
                required: vec![],
            },
            ToolParameter {
                name: "working_directory".to_string(),
                param_type: "string".to_string(),
                description: "Optional directory path where the command should be executed."
                    .to_string(),
                is_required: false,
                enum_values: None,
                items: None,
                properties: None,
                required: vec![],
            },
        ]
    }

    // To correctly specify "command" as a required parameter for the LLM:
    // We might need to override get_json_definition or adjust ToolParameter/base definition.
    // For now, this is a known gap from the FIXME in base.rs.
    // A pragmatic approach for now: the agent's prompt can emphasize which arguments are mandatory.

    #[instrument(skip(self, arguments), fields(tool_name = %self.get_name()))]
    async fn execute(&self, arguments: Value) -> Result<ToolExecResult, ToolError> {
        debug!(args = ?arguments, "Executing bash tool");
        let args: BashToolArgs = serde_json::from_value(arguments.clone()).map_err(|e| {
            ToolError::InvalidArguments {
                tool_name: self.get_name(),
                message: format!("Failed to parse arguments: {}. Expected JSON object with 'command' field. Args: {:?}", e, arguments),
            }
        })?;

        if args.command.trim().is_empty() {
            return Err(ToolError::InvalidArguments {
                tool_name: self.get_name(),
                message: "Command cannot be empty.".to_string(),
            });
        }

        let mut cmd = Command::new("sh");
        cmd.arg("-c");
        cmd.arg(&args.command);
        cmd.stdin(Stdio::null()); // No input to the command
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.kill_on_drop(true); // Ensure process is killed if Child struct is dropped

        if let Some(dir) = &args.working_directory {
            cmd.current_dir(dir);
        }

        debug!(command = %args.command, path = ?args.working_directory, "Configured bash command");

        let child_process_result = cmd.spawn();
        let child = match child_process_result {
            Ok(child) => child,
            Err(e) => {
                error!(error = %e, command = %args.command, "Failed to spawn command");
                return Err(ToolError::ExecutionFailed(format!(
                    "Failed to spawn command '{}': {}",
                    args.command, e
                )));
            }
        };

        let timeout_duration = args.timeout.map(std::time::Duration::from_secs);

        let output_result = if let Some(duration) = timeout_duration {
            match tokio::time::timeout(duration, child.wait_with_output()).await {
                Ok(Ok(output)) => Ok(output),
                Ok(Err(e)) => Err(e), // Inner error from wait_with_output
                Err(_) => {
                    // Timeout elapsed
                    error!(command = %args.command, timeout_sec = duration.as_secs(), "Command timed out");
                    // child will be killed on drop due to kill_on_drop(true)
                    return Ok(ToolExecResult {
                        // Return as a successful tool execution but with timeout info
                        output: None,
                        error: Some(format!(
                            "Command '{}' timed out after {} seconds.",
                            args.command,
                            duration.as_secs()
                        )),
                        error_code: 124, // Common exit code for timeout
                    });
                }
            }
        } else {
            child.wait_with_output().await
        };

        match output_result {
            Ok(output) => {
                let stdout_raw = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr_raw = String::from_utf8_lossy(&output.stderr).to_string();

                let stdout = Self::maybe_truncate(stdout_raw);
                let stderr = Self::maybe_truncate(stderr_raw);

                debug!(stdout_len = stdout.len(), stderr_len = stderr.len(), exit_code = output.status.code(), "Command executed");

                let combined_output = format!("STDOUT:\n{}\nSTDERR:\n{}", stdout, stderr);

                Ok(ToolExecResult {
                    output: Some(combined_output),
                    error: if output.status.success() {
                        None
                    } else {
                        Some(format!(
                            "Command exited with status: {:?}",
                            output.status.code()
                        ))
                    },
                    error_code: output.status.code().unwrap_or(1), // Default to 1 if no exit code (e.g. killed by signal)
                })
            }
            Err(e) => {
                error!(error = %e, command = %args.command, "Command execution failed");
                Err(ToolError::ExecutionFailed(format!(
                    "Command '{}' execution failed: {}",
                    args.command, e
                )))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_bash_tool_echo() {
        let tool = BashTool::new();
        let args = json!({"command": "echo hello"});
        let result = tool.execute(args).await.unwrap();
        assert!(result.output.unwrap().contains("STDOUT:\nhello"));
        assert_eq!(result.error_code, 0);
    }

    #[tokio::test]
    async fn test_bash_tool_error_exit_code() {
        let tool = BashTool::new();
        let args = json!({"command": "exit 123"});
        let result = tool.execute(args).await.unwrap(); // Tool execution itself is successful
        assert!(result.error.is_some());
        assert!(result.error.unwrap().contains("123"));
        assert_eq!(result.error_code, 123);
    }

    #[tokio::test]
    async fn test_bash_tool_stderr() {
        let tool = BashTool::new();
        // This command writes "error message" to stderr and "output message" to stdout
        let args = json!({"command": "echo 'output message'; >&2 echo 'error message'"});
        let result = tool.execute(args).await.unwrap();
        let output_str = result.output.unwrap();
        assert!(output_str.contains("STDOUT:\noutput message"));
        assert!(output_str.contains("STDERR:\nerror message"));
        assert_eq!(result.error_code, 0); // Successful exit code
    }

    #[tokio::test]
    async fn test_bash_tool_invalid_command() {
        let tool = BashTool::new();
        // A command that is likely to not exist or fail due to permissions
        let args = json!({"command": "this_command_should_not_exist_ever_12345"});
        let result = tool.execute(args).await.unwrap(); // Tool execution is successful, command fails
        assert_ne!(result.error_code, 0); // Non-zero exit code
        assert!(result.error.is_some()); // Error message from the tool about exit status
    }

    #[tokio::test]
    async fn test_bash_tool_empty_command_error() {
        let tool = BashTool::new();
        let args = json!({"command": "  "}); // Empty or whitespace only command
        let result = tool.execute(args).await;
        assert!(result.is_err());
        match result.err().unwrap() {
            ToolError::InvalidArguments { message, .. } => {
                assert!(message.contains("Command cannot be empty."));
            }
            _ => panic!("Expected InvalidArguments error for empty command"),
        }
    }

    #[tokio::test]
    async fn test_bash_tool_timeout_success() {
        let tool = BashTool::new();
        let args = json!({"command": "sleep 0.1; echo done", "timeout": 1});
        let result = tool.execute(args).await.unwrap();
        assert!(result.output.unwrap().contains("STDOUT:\ndone"));
        assert_eq!(result.error_code, 0);
    }

    #[tokio::test]
    async fn test_bash_tool_timeout_triggered() {
        let tool = BashTool::new();
        // Command sleeps for 5 seconds, timeout is 1 second
        let args = json!({"command": "sleep 5; echo not_done", "timeout": 1});
        let result = tool.execute(args).await.unwrap(); // Tool execution is successful, command times out

        assert!(result.error.is_some());
        assert!(result.error.unwrap().contains("timed out after 1 seconds"));
        assert_eq!(result.error_code, 124); // Standard timeout error code
        assert!(result.output.is_none()); // No output because it was killed
    }

    #[tokio::test]
    async fn test_bash_tool_working_directory() {
        let tool = BashTool::new();
        // Create a temporary directory and a file in it
        let temp_dir = tempfile::Builder::new()
            .prefix("bash_tool_test")
            .tempdir()
            .unwrap();
        let file_path = temp_dir.path().join("test_file.txt");
        std::fs::write(&file_path, "hello from test file").unwrap();

        let command_to_cat_file =
            format!("cat {}", file_path.file_name().unwrap().to_str().unwrap());
        let args = json!({
            "command": command_to_cat_file,
            "working_directory": temp_dir.path().to_str().unwrap()
        });

        let result = tool.execute(args).await.unwrap();
        assert_eq!(
            result.error_code, 0,
            "Command failed with error: {:?}",
            result.error
        );
        let output = result.output.expect("Expected output from cat command");
        assert!(
            output.contains("STDOUT:\nhello from test file"),
            "Output was: {}",
            output
        );
    }

    #[tokio::test]
    async fn test_bash_tool_output_truncation() {
        let tool = BashTool::new();
        // Create a command that generates a lot of output
        // MAX_BASH_OUTPUT_LEN is 16000. Let's generate more.
        let long_string = "a".repeat(MAX_BASH_OUTPUT_LEN + 1000);
        let command = format!("echo '{}'", long_string);
        let args = json!({ "command": command });

        let result = tool.execute(args).await.unwrap();
        assert_eq!(result.error_code, 0);
        let output = result.output.unwrap();

        assert!(output.contains(TRUNCATED_BASH_MESSAGE));
        // The actual output string length should be MAX_BASH_OUTPUT_LEN,
        // because the original content part + TRUNCATED_BASH_MESSAGE should equal MAX_BASH_OUTPUT_LEN
        // after truncation logic in `maybe_truncate`.
        // The STDOUT:\n and STDERR:\n parts are not counted in MAX_BASH_OUTPUT_LEN for individual streams.
        // So, we check the length of the STDOUT part.

        let stdout_part = output.lines().skip(1) // Skip "STDOUT:"
            .take_while(|line| !line.starts_with("STDERR:")) // Take lines until STDERR
            .collect::<Vec<&str>>().join("\n");

        assert_eq!(stdout_part.len(), MAX_BASH_OUTPUT_LEN);
        assert!(stdout_part.ends_with(TRUNCATED_BASH_MESSAGE));

        // Test with output that, after echo's newline, results in MAX_BASH_OUTPUT_LEN
        // Standard `echo` adds a newline. So, string should be MAX_BASH_OUTPUT_LEN - 1.
        // However, if MAX_BASH_OUTPUT_LEN is 0, this would underflow.
        // Let's assume MAX_BASH_OUTPUT_LEN is at least 1.
        let string_len_for_exact_test = if MAX_BASH_OUTPUT_LEN > 0 { MAX_BASH_OUTPUT_LEN - 1 } else { 0 };
        let exact_limit_string = "b".repeat(string_len_for_exact_test);

        let command_exact = format!("echo '{}'", exact_limit_string);
        let args_exact = json!({"command": command_exact});
        let result_exact = tool.execute(args_exact).await.unwrap();
        assert_eq!(result_exact.error_code, 0, "Exact limit test command failed. Error: {:?}", result_exact.error);
        let output_exact = result_exact.output.unwrap();
        assert!(!output_exact.contains(TRUNCATED_BASH_MESSAGE), "Output at exact limit was unexpectedly truncated. Output: {}", output_exact);

        let stdout_part_exact = output_exact.lines().skip(1).take_while(|line| !line.starts_with("STDERR:")).collect::<Vec<&str>>().join("\n");
        // The length of stdout_part_exact should be the original string length + 1 for newline, if not truncated.
        // If string_len_for_exact_test is 0, echo '' still produces a newline.
        let expected_len_for_exact_stdout = string_len_for_exact_test + if string_len_for_exact_test == 0 && MAX_BASH_OUTPUT_LEN > 0 {1} else if MAX_BASH_OUTPUT_LEN == 0 {0} else {1} ;


        assert_eq!(stdout_part_exact.len(), expected_len_for_exact_stdout, "Stdout part length mismatch for exact limit test.");


        // Test with output much smaller than limit
        let short_string = "c".repeat(100);
        let command_short = format!("echo '{}'", short_string);
        let args_short = json!({"command": command_short});
        let result_short = tool.execute(args_short).await.unwrap();
        assert_eq!(result_short.error_code, 0);
        let output_short = result_short.output.unwrap();
        assert!(!output_short.contains(TRUNCATED_BASH_MESSAGE));
        let stdout_part_short = output_short.lines().skip(1).take_while(|line| !line.starts_with("STDERR:")).collect::<Vec<&str>>().join("\n");
        assert_eq!(stdout_part_short.len(), short_string.len() + 1, "Stdout part length mismatch for short string test."); // +1 for newline
    }
}
