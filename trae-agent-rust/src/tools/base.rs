use crate::llm::base_client as llm_types;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;
use tracing::{debug, error, instrument};

/// Errors that can occur during tool definition or execution.
#[derive(Error, Debug)]
pub enum ToolError {
    /// Indicates that the tool's execution logic failed.
    #[error("Tool execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Invalid arguments for tool '{tool_name}': {message}")]
    InvalidArguments { tool_name: String, message: String },
    #[error("Tool '{0}' not found")]
    NotFound(String), // This is used by EditTool for path validation.
    #[allow(dead_code)] // For future expansion or less common error types
    #[error("Other tool error: {0}")]
    Other(String),
}

/// Represents the direct result of a tool's internal execution logic.
#[derive(Debug, Clone)]
pub struct ToolExecResult {
    /// Optional string output from the tool's successful execution.
    pub output: Option<String>,
    /// Optional error message if the tool's internal logic encountered an issue.
    pub error: Option<String>,
    /// An error code, typically 0 for success and non-zero for failure.
    pub error_code: i32,
}

/// Represents a request to call a specific tool, usually derived from an LLM's tool_call output.
/// This struct is used internally by the `ToolExecutor` before execution.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Currently ToolExecutor uses llm_types::ToolCall directly
pub struct ToolCall {
    /// The name of the tool to be called.
    pub name: String,
    /// The unique ID of this tool call request, as provided by the LLM.
    pub call_id: String,
    /// The arguments for the tool, parsed as a `serde_json::Value`. Expected to be a JSON object.
    pub arguments: Value,
}

/// Represents the final result of a tool execution, formatted for sending back to the LLM.
/// This is what the `ToolExecutor` returns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// The unique ID of the tool call this result corresponds to (mirrors LLM's request).
    pub tool_call_id: String,
    /// Indicates whether the tool execution was successful.
    pub success: bool,
    /// The string output of the tool if successful.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    /// An error message if the tool execution failed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    // TODO: Python version has an 'id' field here too, possibly for OpenAI's specific 'id' for tool message part.
    // If needed, it can be added. For now, this aligns with constructing an LLMMessage of role 'tool'.
}

/// Defines a parameter for a tool, used for generating JSON schema for LLMs.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolParameter {
    /// The name of the parameter.
    pub name: String,
    /// The JSON schema type of the parameter (e.g., "string", "integer", "object", "array").
    #[serde(rename = "type")]
    pub param_type: String,
    /// A description of what the parameter is for.
    pub description: String,
    /// If the parameter type is "string", an optional list of allowed enum values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
    /// If the parameter type is "array", this describes the schema of items in the array.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<ToolParameter>>,
    /// If the parameter type is "object", this provides a map of property names to their `ToolParameter` definitions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, ToolParameter>>,
    /// If the parameter type is "object", this lists the names of properties that are required.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub required: Vec<String>,
    /// Indicates whether this parameter itself is required at the top level for the tool.
    #[serde(default)]
    pub is_required: bool,
}

/// Defines the interface for a tool that can be executed by the agent.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Returns the unique name of the tool.
    fn get_name(&self) -> String;
    /// Returns a description of what the tool does, its purpose, and how to use it.
    fn get_description(&self) -> String;
    /// Returns a list of parameters that the tool accepts.
    fn get_parameters(&self) -> Vec<ToolParameter>;

    /// Executes the tool with the given arguments.
    ///
    /// # Arguments
    /// * `arguments`: A `serde_json::Value` representing the arguments for the tool,
    ///                typically expected to be a JSON object.
    ///
    /// # Returns
    /// A `Result` containing a `ToolExecResult` on success, or a `ToolError` on failure.
    async fn execute(&self, arguments: Value) -> Result<ToolExecResult, ToolError>;

    /// Provides the JSON definition of the tool for the LLM.
    /// This default implementation constructs the schema based on `get_name`, `get_description`,
    /// and `get_parameters`. Tools with very complex input schemas might need to override this.
    fn get_json_definition(&self) -> llm_types::ToolDefinition {
        let mut properties_map: HashMap<String, llm_types::FunctionParameterProperty> =
            HashMap::new();
        let mut required_list = Vec::new();

        for param in self.get_parameters() {
            properties_map.insert(
                param.name.clone(),
                llm_types::FunctionParameterProperty {
                    param_type: param.param_type.clone(),
                    // TODO: Handle complex types like object/array for param.param_type if ToolParameter supports them deeply for schema generation.
                    // Currently, this assumes param_type is a simple JSON type (string, integer, boolean, array - if items is set).
                    // If param.param_type is "object", its `properties` and `required` from ToolParameter are not automatically nested by this default impl.
                    description: param.description.clone(),
                    enum_values: param.enum_values.clone(),
                },
            );
            if param.is_required {
                required_list.push(param.name.clone());
            }
        }

        let llm_fn_params = llm_types::FunctionParameters {
            param_type: "object".to_string(),
            properties: properties_map,
            required: required_list,
        };

        llm_types::ToolDefinition {
            tool_type: "function".to_string(),
            function: llm_types::FunctionDefinition {
                name: self.get_name(),
                description: self.get_description(),
                parameters: llm_fn_params,
            },
        }
    }
}

// Note: The SimpleToolParameter struct and `impl dyn Tool` block were removed as they were intermediate/erroneous.

/// Manages a collection of tools and executes them based on requests from the LLM.
pub struct ToolExecutor {
    tools: HashMap<String, std::sync::Arc<dyn Tool + Send + Sync>>,
}

impl ToolExecutor {
    /// Creates a new `ToolExecutor` with a given list of tools.
    ///
    /// # Arguments
    /// * `tools_list`: A vector of `Arc<dyn Tool>` instances that this executor will manage.
    pub fn new(tools_list: Vec<std::sync::Arc<dyn Tool + Send + Sync>>) -> Self {
        let mut tools = HashMap::new();
        for tool in tools_list {
            tools.insert(tool.get_name(), tool);
        }
        ToolExecutor { tools }
    }

    /// Executes a single tool call request.
    ///
    /// # Arguments
    /// * `tool_call_request`: An `llm_types::ToolCall` struct representing the LLM's request to call a tool.
    ///
    /// # Returns
    /// A `ToolResult` to be sent back to the LLM.
    #[instrument(skip(self, tool_call_request), fields(tool_name = %tool_call_request.function.name))]
    pub async fn execute_tool_call(&self, tool_call_request: &llm_types::ToolCall) -> ToolResult {
        debug!(args = %tool_call_request.function.arguments, "Attempting to execute tool");
        match self.tools.get(&tool_call_request.function.name) {
            Some(tool) => {
                match serde_json::from_str::<Value>(&tool_call_request.function.arguments) {
                    Ok(args_value) => {
                        // Arguments are considered valid if they parse to a JSON object or JSON null.
                        // Empty argument string is handled in the Err arm and results in Value::Null.
                        if !args_value.is_object() && !args_value.is_null() {
                            error!(args = %tool_call_request.function.arguments, "Parsed tool arguments are not a JSON object or JSON null.");
                            return ToolResult {
                                tool_call_id: tool_call_request.id.clone(),
                                success: false,
                                result: None,
                                error: Some(format!("Tool arguments must parse to a JSON object or null. Parsed as: {}", args_value)),
                            };
                        }
                        match tool.execute(args_value).await {
                            Ok(exec_result) => ToolResult {
                                tool_call_id: tool_call_request.id.clone(),
                                success: exec_result.error_code == 0,
                                result: exec_result.output,
                                error: exec_result.error,
                            },
                            Err(e) => {
                                error!(error = %e, tool_name = %tool.get_name(), "Tool execution failed");
                                ToolResult {
                                    tool_call_id: tool_call_request.id.clone(),
                                    success: false,
                                    result: None,
                                    error: Some(e.to_string()),
                                }
                            }
                        }
                    }
                    Err(e) => {
                        // Handle case where arguments string might be empty, which is valid for tools with no args
                        if tool_call_request.function.arguments.trim().is_empty() {
                            match tool.execute(Value::Null).await {
                                // Pass Value::Null for empty args
                                Ok(exec_result) => ToolResult {
                                    tool_call_id: tool_call_request.id.clone(),
                                    success: exec_result.error_code == 0,
                                    result: exec_result.output,
                                    error: exec_result.error,
                                },
                                Err(tool_err) => {
                                    error!(error = %tool_err, tool_name = %tool.get_name(), "Tool execution failed with empty args");
                                    ToolResult {
                                        tool_call_id: tool_call_request.id.clone(),
                                        success: false,
                                        result: None,
                                        error: Some(tool_err.to_string()),
                                    }
                                }
                            }
                        } else {
                            error!(args = %tool_call_request.function.arguments, error = %e, "Failed to parse tool arguments");
                            ToolResult {
                                tool_call_id: tool_call_request.id.clone(),
                                success: false,
                                result: None,
                                error: Some(format!(
                                    "Invalid JSON arguments for tool {}: {}. Arguments: {}",
                                    tool_call_request.function.name,
                                    e,
                                    tool_call_request.function.arguments
                                )),
                            }
                        }
                    }
                }
            }
            None => {
                error!(tool_name = %tool_call_request.function.name, "Tool not found");
                ToolResult {
                    tool_call_id: tool_call_request.id.clone(),
                    success: false,
                    result: None,
                    error: Some(format!(
                        "Tool '{}' not found. Available tools: {:?}",
                        tool_call_request.function.name,
                        self.tools.keys()
                    )),
                }
            }
        }
    }

    /// Executes multiple tool calls in parallel.
    /// TODO: Not currently used by BaseAgent, which does sequential calls. Can be used if parallel execution is desired.
    #[allow(dead_code)]
    pub async fn parallel_tool_calls(&self, tool_calls: &[llm_types::ToolCall]) -> Vec<ToolResult> {
        let futures = tool_calls.iter().map(|call| self.execute_tool_call(call));
        futures::future::join_all(futures).await
    }

    /// Executes multiple tool calls sequentially.
    pub async fn sequential_tool_calls(
        &self,
        tool_calls: &[llm_types::ToolCall],
    ) -> Vec<ToolResult> {
        let mut results = Vec::new();
        for call in tool_calls {
            results.push(self.execute_tool_call(call).await);
        }
        results
    }
}
