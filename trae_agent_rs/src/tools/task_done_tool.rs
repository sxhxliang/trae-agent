use super::base::{Tool, ToolError, ToolExecResult, ToolParameter};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use tracing::info;

#[derive(Deserialize, Debug)]
struct TaskDoneArgs {
    summary: Option<String>, // Optional summary of the work done
                             // Add other relevant fields like status (success/failure details) if needed
}

pub struct TaskDoneTool;

impl TaskDoneTool {
    pub fn new() -> Self {
        TaskDoneTool
    }
}

#[async_trait] // Added
impl Tool for TaskDoneTool {
    fn get_name(&self) -> String {
        "task_done".to_string()
    }

    fn get_description(&self) -> String {
        "Signals that the current task is considered complete by the agent. \
        Call this when you are confident the objectives have been met. \
        Optionally provide a summary of the work."
            .to_string()
    }

    fn get_parameters(&self) -> Vec<ToolParameter> {
        // As per FIXME in tools/base.rs, the default get_json_definition might not correctly list `summary` as non-required.
        // For now, the agent's prompt should guide its usage.
        vec![
            ToolParameter {
                name: "summary".to_string(),
                param_type: "string".to_string(),
                description:
                    "An optional summary of what was achieved or the final state of the task."
                        .to_string(),
                is_required: false, // Mark as not required
                enum_values: None,
                items: None,
                properties: None,
                required: vec![],
            }, // If other parameters like 'status_code' or 'patch_content' were part of this, they'd be defined here.
        ]
    }

    // To make the `summary` parameter explicitly optional for the LLM,
    // the `get_json_definition` in `Tool` trait or a specific override here would need to ensure
    // "summary" is not in the `required` list of the top-level parameters object.
    // The current default `get_json_definition` has a FIXME for this.

    async fn execute(&self, arguments: Value) -> Result<ToolExecResult, ToolError> {
        info!(args = ?arguments, tool_name = %self.get_name(), "Executing task_done tool");

        let args: TaskDoneArgs =
            serde_json::from_value(arguments.clone()).map_err(|e| ToolError::InvalidArguments {
                tool_name: self.get_name(),
                message: format!("Failed to parse arguments: {}. Args: {:?}", e, arguments),
            })?;

        let summary_message = args
            .summary
            .unwrap_or_else(|| "No summary provided.".to_string());

        // The primary purpose of this tool is to be recognized by the agent's control flow.
        // The output can confirm its invocation.
        Ok(ToolExecResult {
            output: Some(format!(
                "Task completion signaled. Summary: {}",
                summary_message
            )),
            error: None,
            error_code: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_task_done_tool_with_summary() {
        let tool = TaskDoneTool::new();
        let args = json!({"summary": "Successfully completed all objectives."});
        let result = tool.execute(args).await.unwrap();
        assert!(result
            .output
            .unwrap()
            .contains("Summary: Successfully completed all objectives."));
        assert_eq!(result.error_code, 0);
    }

    #[tokio::test]
    async fn test_task_done_tool_without_summary() {
        let tool = TaskDoneTool::new();
        let args = json!({}); // No summary argument
        let result = tool.execute(args).await.unwrap();
        assert!(result
            .output
            .unwrap()
            .contains("Summary: No summary provided."));
        assert_eq!(result.error_code, 0);
    }

    #[tokio::test]
    async fn test_task_done_tool_empty_summary() {
        let tool = TaskDoneTool::new();
        let args = json!({"summary": ""});
        let result = tool.execute(args).await.unwrap();
        assert!(result.output.unwrap().contains("Summary: ")); // Summary is empty string
        assert_eq!(result.error_code, 0);
    }
}
