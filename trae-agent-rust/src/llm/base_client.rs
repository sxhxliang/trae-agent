use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

// Re-export ModelParameters from config and alias it for clarity within LLM context if needed,
// or define a specific one here if it diverges. For now, assume config::ModelParameters is sufficient.
pub use crate::config::ModelParameters; // Re-exporting for convenience within the LLM module context.

/// Errors that can occur during LLM client operations.
#[derive(Error, Debug)]
pub enum LLMError {
    /// Error related to network requests (e.g., connection refused, timeout).
    #[error("HTTP request failed: {0}")]
    Network(reqwest::Error),
    /// Error reported by the LLM API (e.g., invalid request, rate limit).
    #[error("API error: {0}")]
    ApiError(String),
    /// Error during parsing of the LLM's response (e.g., malformed JSON).
    #[allow(dead_code)] // May become used with more complex error handling or other clients
    #[error("Failed to parse response: {0}")]
    ParsingError(serde_json::Error),
    /// Required API key was not provided.
    #[error("No API key provided")]
    NoApiKey,
    /// The specified model or provider is not supported by the client.
    #[allow(dead_code)] // Currently checked in BaseAgent::try_new, but could be checked by client itself
    #[error("Unsupported model or provider")]
    UnsupportedModel,
    /// Any other type of error.
    #[error("Other error: {0}")]
    Other(String),
}

/// Represents the role of a message in a conversation.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System message, usually setting context or instructions for the LLM.
    System,
    /// Message from the end-user.
    User,
    /// Message from the AI assistant.
    Assistant,
    /// Message representing the result of a tool execution.
    Tool,
}

/// Represents a single message in a conversation with an LLM.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LLMMessage {
    /// The role of the message sender.
    pub role: MessageRole,
    /// The textual content of the message. Optional, as some messages (e.g., assistant responses with tool calls) might not have direct text content.
    pub content: Option<String>,
    /// Optional name of the sender or tool, used for specific roles like 'tool'.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional list of tool calls requested by the assistant.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Optional ID of the tool call, used when this message is a response from a tool.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// Describes the function called by an LLM tool request.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCallFunction {
    /// The name of the function to be called.
    pub name: String,
    /// A JSON string representing the arguments to the function.
    pub arguments: String,
}

/// Represents a tool call requested by the LLM.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolCall {
    /// A unique identifier for this specific tool call.
    pub id: String,
    /// The type of the tool, typically "function".
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function and arguments for the tool call.
    pub function: ToolCallFunction,
}

/// Specifies a tool the LLM should call, used in requests (OpenAI specific).
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolChoice {
    /// The type of the tool, typically "function".
    #[serde(rename = "type")]
    pub tool_type: String,
    /// Specifies the function to be called.
    pub function: ToolChoiceFunction,
}

/// Specifies the function name for a forced tool choice.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolChoiceFunction {
    /// The name of the function to force call.
    pub name: String,
}

/// Defines a tool (specifically a function) that the LLM can call.
/// This structure is typically part of the request to the LLM.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolDefinition {
    /// The type of the tool, typically "function".
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The definition of the function.
    pub function: FunctionDefinition,
}

/// Defines the structure of a function that can be called by the LLM.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionDefinition {
    /// The name of the function.
    pub name: String,
    /// A description of what the function does.
    pub description: String,
    /// The parameters the function accepts, described as a JSON schema.
    pub parameters: FunctionParameters,
}

/// Describes the parameters of a function, conforming to JSON Schema.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionParameters {
    /// The type of the parameters object, typically "object".
    #[serde(rename = "type")]
    pub param_type: String,
    /// A map of property names to their definitions.
    pub properties: HashMap<String, FunctionParameterProperty>,
    /// A list of names of required properties.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub required: Vec<String>,
}

/// Defines a single property within a function's parameters.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FunctionParameterProperty {
    /// The JSON schema type of the property (e.g., "string", "integer", "boolean").
    #[serde(rename = "type")]
    pub param_type: String,
    /// A description of the property.
    pub description: String,
    /// Optional list of allowed enum values for a string property.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enum_values: Option<Vec<String>>,
}

/// Represents a single choice (among potentially multiple, e.g., with n > 1) from an LLM response.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LLMResponseChoice {
    /// The index of this choice in the list of choices.
    pub index: u32,
    /// The message generated by the LLM for this choice.
    pub message: LLMMessage,
    /// The reason the LLM finished generating tokens (e.g., "stop", "tool_calls", "length").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// Represents token usage information for an LLM API call.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LLMUsage {
    /// Number of tokens in the prompt.
    pub prompt_tokens: u32,
    /// Number of tokens in the completion (response). Optional.
    pub completion_tokens: Option<u32>,
    /// Total number of tokens used in the request (prompt + completion).
    pub total_tokens: u32,
}

/// Represents the overall response from an LLM chat completion API.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LLMResponse {
    /// A unique identifier for the LLM response.
    pub id: String,
    /// The type of object, typically "chat.completion".
    pub object: String,
    /// Unix timestamp of when the response was created.
    pub created: u64,
    /// The model that generated the response.
    pub model: String,
    /// A list of choices generated by the LLM. Usually one, unless `n` > 1 was requested.
    pub choices: Vec<LLMResponseChoice>,
    /// Optional token usage information for the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<LLMUsage>,
    // TODO: Consider how to best represent provider-specific fields if they diverge significantly.
    // For Anthropic, content is directly an array of ContentBlock
    // and stop_reason is top-level. This struct is more OpenAI-like.
    // A more generic internal representation or provider-specific response structs might be needed.
}

/// Represents the result of a tool execution, formatted for inclusion in an LLM message.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ToolResult {
    /// The ID of the tool call this result corresponds to.
    pub tool_call_id: String,
    /// The role for this message, should be "tool".
    pub role: MessageRole,
    /// The name of the tool that was called.
    pub name: String,
    /// The string content result of the tool execution.
    pub content: String,
}

/// Trait defining the interface for an LLM client.
///
/// This allows for different LLM providers (OpenAI, Anthropic, etc.) to be used
/// interchangeably by the agent.
#[async_trait]
pub trait LLMClient: Send + Sync {
    /// Creates a new instance of the LLM client.
    ///
    /// # Arguments
    /// * `api_key`: Optional API key. If not provided, client might try to load from environment.
    /// * `base_url`: Optional custom base URL for the API.
    /// * `model_parameters`: Default model parameters to use for this client instance.
    async fn new(
        api_key: Option<String>,
        base_url: Option<String>,
        model_parameters: ModelParameters,
    ) -> Result<Self, LLMError>
    where
        Self: Sized;

    /// Sends a chat request to the LLM.
    ///
    /// # Arguments
    /// * `messages`: A list of `LLMMessage` representing the conversation history.
    /// * `tools`: Optional list of `ToolDefinition` that the LLM can choose to call.
    /// * `tool_choice`: Optional mechanism to force the LLM to call a specific tool (OpenAI specific).
    ///
    /// # Returns
    /// A `Result` containing the `LLMResponse` or an `LLMError`.
    async fn chat(
        &self,
        messages: Vec<LLMMessage>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>, // Added for OpenAI
    ) -> Result<LLMResponse, LLMError>;

    // Optional: A method to get provider name
    // fn provider_name(&self) -> &'static str;
}
