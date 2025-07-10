use super::base::{Tool, ToolError, ToolExecResult, ToolParameter};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use tracing::{debug, info, instrument};

#[derive(Deserialize, Debug)]
struct SequentialThinkingArgs {
    thought: String,
    thought_number: u32,
    total_thoughts: u32,
    next_thought_needed: bool,
    is_revision: Option<bool>,
    revises_thought: Option<u32>,
    branch_from_thought: Option<u32>,
    branch_id: Option<String>,
    needs_more_thoughts: Option<bool>,
}

pub struct SequentialThinkingTool;

impl SequentialThinkingTool {
    pub fn new() -> Self {
        SequentialThinkingTool
    }
}

#[async_trait]
impl Tool for SequentialThinkingTool {
    fn get_name(&self) -> String {
        "sequential_thinking".to_string()
    }

    fn get_description(&self) -> String {
        // Adapted from Python's TraeAgent system prompt guidance
        "A tool for dynamic and reflective problem-solving through a sequence of thoughts. \
        Each thought can build on, question, or revise previous insights. \
        Use to break down complex problems, plan steps, perform analysis, generate and verify hypotheses. \
        The LLM provides the content of each thought along with metadata about its place in the sequence."
            .to_string()
    }

    fn get_parameters(&self) -> Vec<ToolParameter> {
        vec![
            ToolParameter {
                name: "thought".to_string(), param_type: "string".to_string(),
                description: "Your current thinking step, including analysis, hypotheses, or revisions.".to_string(),
                is_required: true, enum_values: None, items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "thought_number".to_string(), param_type: "integer".to_string(),
                description: "Current thought number in the sequence (must be >= 1).".to_string(),
                is_required: true, enum_values: None, items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "total_thoughts".to_string(), param_type: "integer".to_string(),
                description: "Current estimated total number of thoughts needed for this sequence (must be >= 1). Can be adjusted by the LLM.".to_string(),
                is_required: true, enum_values: None, items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "next_thought_needed".to_string(), param_type: "boolean".to_string(),
                description: "Set to true if more thinking steps are planned in this sequence, false if this is the final thought for now.".to_string(),
                is_required: true, enum_values: None, items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "is_revision".to_string(), param_type: "boolean".to_string(),
                description: "Optional: true if this thought revises or questions a previous thought.".to_string(),
                is_required: false, enum_values: None, items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "revises_thought".to_string(), param_type: "integer".to_string(),
                description: "Optional: If is_revision is true, the number of the thought being revised (must be >= 1).".to_string(),
                is_required: false, enum_values: None, items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "branch_from_thought".to_string(), param_type: "integer".to_string(),
                description: "Optional: If this thought starts a new branch of thinking, the number of the thought it branches from (must be >= 1).".to_string(),
                is_required: false, enum_values: None, items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "branch_id".to_string(), param_type: "string".to_string(),
                description: "Optional: An identifier for the current branch of thinking, if applicable.".to_string(),
                is_required: false, enum_values: None, items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "needs_more_thoughts".to_string(), param_type: "boolean".to_string(),
                description: "Optional: Set to true if, upon reaching 'total_thoughts', you realize more thoughts are needed in the current sequence/branch.".to_string(),
                is_required: false, enum_values: None, items: None, properties: None, required: vec![],
            },
        ]
    }

    #[instrument(skip(self, arguments), fields(tool_name = %self.get_name()))]
    async fn execute(&self, arguments: Value) -> Result<ToolExecResult, ToolError> {
        debug!(args = ?arguments, "Executing sequential_thinking tool");
        let args: SequentialThinkingArgs =
            serde_json::from_value(arguments.clone()).map_err(|e| ToolError::InvalidArguments {
                tool_name: self.get_name(),
                message: format!("Failed to parse arguments: {}. Args: {:?}", e, arguments),
            })?;

        if args.thought_number < 1 {
            return Err(ToolError::InvalidArguments {
                tool_name: self.get_name(),
                message: "thought_number must be at least 1.".to_string(),
            });
        }
        if args.total_thoughts < 1 {
            return Err(ToolError::InvalidArguments {
                tool_name: self.get_name(),
                message: "total_thoughts must be at least 1.".to_string(),
            });
        }
        if let Some(rev_t) = args.revises_thought {
            if rev_t < 1 {
                return Err(ToolError::InvalidArguments {
                    tool_name: self.get_name(),
                    message: "revises_thought must be at least 1.".to_string(),
                });
            }
        }
        if let Some(branch_t) = args.branch_from_thought {
            if branch_t < 1 {
                return Err(ToolError::InvalidArguments {
                    tool_name: self.get_name(),
                    message: "branch_from_thought must be at least 1.".to_string(),
                });
            }
        }

        let mut output_parts = vec![format!(
            "Thought {}/{}: {}",
            args.thought_number, args.total_thoughts, args.thought
        )];
        if args.is_revision.unwrap_or(false) {
            output_parts.push(format!(
                "(Revises thought {})",
                args.revises_thought.unwrap_or(0)
            ));
        }
        if args.branch_id.is_some() || args.branch_from_thought.is_some() {
            output_parts.push(format!(
                "(Branch: {} from thought {})",
                args.branch_id.as_deref().unwrap_or("N/A"),
                args.branch_from_thought.unwrap_or(0)
            ));
        }
        if args.next_thought_needed {
            output_parts.push("(Next thought needed)".to_string());
        } else {
            output_parts.push("(Sequence concludes for now)".to_string());
        }
        if args.needs_more_thoughts.unwrap_or(false) {
            output_parts.push(
                "(More thoughts needed in this sequence despite reaching initial total)"
                    .to_string(),
            );
        }

        let formatted_output = output_parts.join(" - ");
        info!("LLM recorded thought: {}", formatted_output);

        // The Python tool's output is a JSON string of a dict like:
        // {"thought_number": ..., "total_thoughts": ..., "next_thought_needed": ..., ...}
        // Let's try to match that for better compatibility if the agent expects it.
        let response_data = serde_json::json!({
            "thought_recorded": args.thought,
            "thought_number": args.thought_number,
            "total_thoughts": args.total_thoughts,
            "next_thought_needed": args.next_thought_needed,
            "is_revision": args.is_revision,
            "revises_thought": args.revises_thought,
            "branch_from_thought": args.branch_from_thought,
            "branch_id": args.branch_id,
            "needs_more_thoughts": args.needs_more_thoughts,
            "status_message": formatted_output // Include the human-readable summary too
        });

        Ok(ToolExecResult {
            output: Some(
                serde_json::to_string_pretty(&response_data).unwrap_or_else(|_| formatted_output),
            ),
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
    async fn test_sequential_thinking_tool_simple_thought() {
        let tool = SequentialThinkingTool::new();
        let args = json!({
            "thought": "This is the first step of my plan.",
            "thought_number": 1,
            "total_thoughts": 3,
            "next_thought_needed": true
        });
        let result = tool.execute(args).await.unwrap();
        assert!(result.output.is_some());
        let output_val: Value = serde_json::from_str(&result.output.unwrap()).unwrap();

        assert_eq!(output_val["thought_number"], 1);
        assert_eq!(
            output_val["thought_recorded"],
            "This is the first step of my plan."
        );
        assert_eq!(output_val["next_thought_needed"], true);
        assert!(output_val["status_message"]
            .as_str()
            .unwrap()
            .contains("Thought 1/3: This is the first step of my plan."));
        assert!(output_val["status_message"]
            .as_str()
            .unwrap()
            .contains("(Next thought needed)"));
        assert_eq!(result.error_code, 0);
    }

    #[tokio::test]
    async fn test_sequential_thinking_tool_revision() {
        let tool = SequentialThinkingTool::new();
        let args = json!({
            "thought": "Revising my approach to step 1.",
            "thought_number": 2,
            "total_thoughts": 3,
            "next_thought_needed": true,
            "is_revision": true,
            "revises_thought": 1
        });
        let result = tool.execute(args).await.unwrap();
        assert!(result.output.is_some());
        let output_val: Value = serde_json::from_str(&result.output.unwrap()).unwrap();
        assert_eq!(output_val["is_revision"], true);
        assert_eq!(output_val["revises_thought"], 1);
        assert!(output_val["status_message"]
            .as_str()
            .unwrap()
            .contains("(Revises thought 1)"));
    }

    #[tokio::test]
    async fn test_sequential_thinking_tool_invalid_thought_number() {
        let tool = SequentialThinkingTool::new();
        let args = json!({
            "thought": "Invalid.",
            "thought_number": 0, // Invalid
            "total_thoughts": 3,
            "next_thought_needed": true
        });
        let result = tool.execute(args).await;
        assert!(result.is_err());
        match result.err().unwrap() {
            ToolError::InvalidArguments { message, .. } => {
                assert!(message.contains("thought_number must be at least 1"));
            }
            _ => panic!("Expected InvalidArguments error"),
        }
    }

    #[tokio::test]
    async fn test_sequential_thinking_tool_missing_required_arg() {
        let tool = SequentialThinkingTool::new();
        let args = json!({
            "thought_number": 1,
            "total_thoughts": 1,
            "next_thought_needed": false
            // "thought" is missing
        });
        let result = tool.execute(args).await;
        assert!(result.is_err());
        match result.err().unwrap() {
            ToolError::InvalidArguments { message, .. } => {
                assert!(message.contains("Failed to parse arguments")); // Serde will complain about missing 'thought'
            }
            _ => panic!("Expected InvalidArguments error for missing 'thought'"),
        }
    }
}
