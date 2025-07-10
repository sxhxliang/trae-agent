//! # Tools Module
//!
//! Defines the framework for tools that the agent can use, including a `Tool` trait,
//! a `ToolExecutor` for running tools, and a `ToolRegistry` for managing available tools.
//! Concrete tool implementations like `BashTool`, `EditTool`, etc., are also part of this module.

pub mod base;
pub mod bash_tool;
pub mod edit_tool;
pub mod json_edit_tool; // Added
pub mod sequential_thinking_tool;
pub mod task_done_tool;

pub use base::{Tool, ToolError, ToolExecutor, ToolResult as AgentToolResult};
pub use bash_tool::BashTool;
pub use edit_tool::EditTool;
pub use json_edit_tool::JsonEditTool; // Added
pub use sequential_thinking_tool::SequentialThinkingTool;
pub use task_done_tool::TaskDoneTool;

use std::collections::HashMap;
use std::sync::Arc;

/// A registry for discovering and managing available tools.
///
/// Tools are registered by name, allowing the agent to look them up
/// and get their definitions for LLM interaction.
// TODO: Potentially load tools dynamically or from configuration in the future.
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool + Send + Sync>>,
}

impl ToolRegistry {
    /// Creates a new, empty `ToolRegistry`.
    pub fn new() -> Self {
        ToolRegistry {
            tools: HashMap::new(),
        }
    }

    /// Registers a tool with the registry.
    ///
    /// # Arguments
    /// * `tool`: An instance of a type implementing the `Tool` trait.
    pub fn register<T: Tool + Send + Sync + 'static>(&mut self, tool: T) {
        self.tools.insert(tool.get_name(), Arc::new(tool));
    }

    /// Retrieves a tool by its name.
    ///
    /// # Arguments
    /// * `name`: The name of the tool to retrieve.
    ///
    /// # Returns
    /// An `Option` containing an `Arc` to the tool if found, otherwise `None`.
    #[allow(dead_code)] // May be useful for direct tool inspection or invocation
    pub fn get_tool(&self, name: &str) -> Option<Arc<dyn Tool + Send + Sync>> {
        self.tools.get(name).cloned()
    }

    /// Gets the JSON definitions of all registered tools, for use with LLMs.
    pub fn get_all_tool_definitions(&self) -> Vec<crate::llm::base_client::ToolDefinition> {
        self.tools
            .values()
            .map(|tool| tool.get_json_definition())
            .collect()
    }

    /// Gets `Arc` references to all registered tools.
    pub fn get_all_tools_arc(&self) -> Vec<Arc<dyn Tool + Send + Sync>> {
        self.tools.values().cloned().collect()
    }
}

impl Default for ToolRegistry {
    /// Creates a `ToolRegistry` populated with default tools.
    fn default() -> Self {
        let mut registry = Self::new();
        // Register default tools here
        registry.register(BashTool::new());
        registry.register(EditTool::new());
        registry.register(JsonEditTool::new()); // Added
        registry.register(SequentialThinkingTool::new());
        registry.register(TaskDoneTool::new());
        registry
    }
}

/// Helper function to generate LLM tool definitions from a list of tool instances.
///
/// This can be useful if tools are managed outside of a central `ToolRegistry`.
#[allow(dead_code)] // This is a utility function that might be useful later but not currently called.
pub fn llm_tool_definitions_from_tools(
    tools_list: &[Arc<dyn Tool + Send + Sync>],
) -> Vec<crate::llm::base_client::ToolDefinition> {
    tools_list
        .iter()
        .map(|tool| tool.get_json_definition())
        .collect()
}
