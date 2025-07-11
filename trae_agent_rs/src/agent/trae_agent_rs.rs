use super::base_agent::{common_execute_task_loop, Agent, AgentError, AgentEvent, BaseAgent};
use crate::config::Config;
use crate::llm::base_client::{LLMMessage, LLMResponse, MessageRole};
use crate::tools::ToolRegistry;
use crate::utils::trajectory_recorder::TrajectoryRecorder; // Added
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap; // Added
use std::path::PathBuf; // Added
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// The Trae Agent implementation, specialized for software engineering tasks.
///
/// This agent uses a `BaseAgent` for common functionalities and implements
/// specific logic for task setup, system prompts, and determining task completion
/// (including patch validation if required).
pub struct TraeAgent {
    base_agent: BaseAgent,
}

impl TraeAgent {
    /// Attempts to create a new `TraeAgent`.
    ///
    /// Initializes the underlying `BaseAgent` with the given configuration and tool registry.
    ///
    /// # Arguments
    /// * `config`: Shared application configuration.
    /// * `tool_registry`: Shared registry of available tools.
    ///
    /// # Returns
    /// A `Result` containing the new `TraeAgent` or an `AgentError` if initialization fails.
    pub async fn try_new(
        config: Arc<Config>,
        tool_registry: Arc<ToolRegistry>,
        trajectory_file_path: Option<PathBuf>, // Added
    ) -> Result<Self, AgentError> {
        let mut base_agent = BaseAgent::try_new(config.clone(), tool_registry).await?; // Cloned config for recorder

        if let Some(path) = trajectory_file_path {
            match TrajectoryRecorder::new(Some(path)) {
                Ok(recorder) => {
                    base_agent.set_trajectory_recorder(recorder);
                    info!("Trajectory recorder initialized for TraeAgent.");
                }
                Err(e) => {
                    warn!("Failed to initialize TrajectoryRecorder: {}. Continuing without trajectory recording.", e);
                    // Optionally, return an error if trajectory recording is critical
                    // return Err(AgentError::ConfigError(format!("Failed to init trajectory recorder: {}", e)));
                }
            }
        }

        Ok(Self { base_agent })
    }


    /// Generates the system prompt specific to the `TraeAgent`.
    /// This prompt instructs the LLM on its role as a software engineering agent.
    fn get_system_prompt(&self) -> String {
        // TODO: Consider loading this prompt from a configuration file or template
        // for easier modification and versioning.
        "You are an expert AI software engineering agent. \
        Your primary goal is to resolve a given GitHub issue by navigating the provided codebase, \
        identifying the root cause of the bug, implementing a robust fix, and ensuring your changes are safe and well-tested. \
        Use the provided tools to interact with the file system, run commands, and manage your thought process. \
        If you are sure the issue has been solved, you should use a specific mechanism or tool (e.g., 'task_done' if available) to indicate completion.".to_string()
    }

    /// Determines if the agent should stop execution based on the LLM response and task parameters.
    ///
    /// This static method includes logic for:
    /// - Checking if maximum steps have been reached.
    /// - Detecting if the `task_done` tool was called by the LLM.
    /// - If `must_patch` is true and `task_done` was called, validating that a non-empty patch
    ///   (excluding test files) was generated.
    ///
    /// # Returns
    /// A `StopReason` enum indicating why the agent should stop or if it should continue.
    fn fn_should_stop(
        llm_response: &LLMResponse,
        current_step_number: u32,
        max_steps: u32,
        must_patch: bool,
        project_path: Option<&str>,
        base_commit: Option<&str>,
    ) -> super::base_agent::StopReason { // Changed return type
        if current_step_number >= max_steps {
            warn!("Max steps reached, forcing stop.");
            return super::base_agent::StopReason::MaxStepsReached;
        }

        let mut completion_signaled_by_tool = false;
        // Check for task_done tool call
        if let Some(tool_calls) = &llm_response.choices[0].message.tool_calls {
            if tool_calls.iter().any(|tc| tc.function.name == "task_done") {
                info!("'task_done' tool called by LLM.");
                completion_signaled_by_tool = true;
            }
        }

        // Check for textual completion cues (Python's llm_indicates_task_completed)
        let mut completion_signaled_by_text = false;
        if let Some(content) = &llm_response.choices[0].message.content {
            let content_lower = content.to_lowercase();
            let completion_indicators = [
                "task completed",
                "task finished",
                "done",
                "completed successfully",
                "finished successfully",
            ];
            if completion_indicators.iter().any(|indicator| content_lower.contains(indicator)) {
                info!("Textual completion cue detected in LLM response.");
                completion_signaled_by_text = true;
            }
        }

        if completion_signaled_by_tool || completion_signaled_by_text {
            // If must_patch is true, validate the patch
            if must_patch {
                if let Some(proj_p) = project_path {
                    match crate::utils::git_utils::get_git_diff(proj_p, base_commit) {
                        Ok(model_patch) => {
                            let patch = crate::utils::git_utils::remove_patches_to_tests(&model_patch);
                            if patch.trim().is_empty() {
                                warn!("Completion signaled (tool or text), but 'must_patch' is true and generated patch is empty. Task not considered done.");
                                return super::base_agent::StopReason::ValidationFailed(
                                    "ERROR! Your Patch is empty. Please provide a patch that fixes the problem.".to_string(),
                                );
                            }
                            info!("Patch validation successful for completion signal.");
                            return super::base_agent::StopReason::TaskCompleted;
                        }
                        Err(e) => {
                            error!("Failed to get git diff for patch validation: {}", e);
                            return super::base_agent::StopReason::ValidationFailed(format!(
                                "ERROR! Could not verify patch due to git diff error: {}. Please try the fix again.",
                                e
                            ));
                        }
                    }
                } else {
                    warn!("'must_patch' is true, but no project_path is available for git diff. Assuming task not complete.");
                    return super::base_agent::StopReason::ValidationFailed(
                        "ERROR! 'must_patch' is true, but project_path is not configured for diffing.".to_string(),
                    );
                }
            } else {
                // must_patch is false, so tool call or text cue is enough
                return super::base_agent::StopReason::TaskCompleted;
            }
        }

        super::base_agent::StopReason::Continue // Default: don't stop
    }

    /// Processes the LLM's response to extract a final message when the task is considered complete.
    ///
    /// This static method is typically called when `fn_should_stop` indicates completion without
    /// a validation error. It checks for arguments to the `task_done` tool (e.g., a summary)
    /// or defaults to the LLM's main content.
    fn fn_process_llm_response_for_completion(llm_response: &LLMResponse) -> Option<String> {
        if let Some(tool_calls) = &llm_response.choices[0].message.tool_calls {
            for tc in tool_calls {
                if tc.function.name == "task_done" {
                    if let Ok(args_val) = serde_json::from_str::<Value>(&tc.function.arguments) {
                        if let Some(summary) = args_val.get("summary").and_then(|s| s.as_str()) {
                            return Some(format!("Task completed. Summary: {}", summary));
                        }
                    }
                    return Some("Task marked as done by the agent.".to_string());
                }
            }
        }
        llm_response.choices[0].message.content.clone()
    }

    // --- Methods for Interactive Mode ---

    /// Determines if an interactive turn should stop.
    /// Stops if:
    /// 1. LLM provides a direct textual response (no tool calls).
    /// 2. After one round of tool calls, the LLM provides any response.
    /// 3. Max steps for the turn (e.g., 2-3) are reached.
    fn fn_should_stop_interactive(
        llm_response: &LLMResponse,
        current_step_in_turn: u32, // This is the step number within common_execute_task_loop
        max_steps_for_turn: u32,   // Max steps for this specific interactive turn (e.g., 2)
    ) -> super::base_agent::StopReason { // Changed return type
        if current_step_in_turn >= max_steps_for_turn {
            return super::base_agent::StopReason::TaskCompleted; // Turn considered "complete" by reaching its micro-step limit
        }

        let message = &llm_response.choices[0].message;
        if message.tool_calls.is_none()
            || message.tool_calls.as_ref().map_or(true, |tc| tc.is_empty())
        {
            // If there are no tool calls, this is a direct response to the user, so the turn is "complete".
            return super::base_agent::StopReason::TaskCompleted;
        }

        // If it's the first step (current_step_in_turn == 1) and there ARE tool calls, we want to continue.
        if current_step_in_turn == 1 && message.tool_calls.is_some() && !message.tool_calls.as_ref().unwrap().is_empty() {
            return super::base_agent::StopReason::Continue;
        }

        // If it's past the first step (e.g., current_step_in_turn == 2, meaning tools were just run from step 1),
        // then this LLM response is the one after tool execution. The turn should be considered "complete".
        // This also covers cases where max_steps_for_turn might be > 2, but we typically want to stop after one round of tool calls.
        // The check for current_step_in_turn >= max_steps_for_turn already handles the absolute limit.
        if current_step_in_turn > 1 { // Implicitly, this means tools were likely called in a previous micro-step
            return super::base_agent::StopReason::TaskCompleted;
        }

        // Default for any other scenario (though the logic above should cover typical interactive turn patterns)
        super::base_agent::StopReason::Continue
    }

    /// Processes the LLM's response for an interactive turn to get the assistant's message.
    fn fn_extract_assistant_response_interactive(llm_response: &LLMResponse) -> Option<String> {
        // For interactive mode, the "final result" is simply the assistant's content.
        llm_response.choices[0].message.content.clone()
    }

    /// Saves the git diff to the specified patch_path if configured.
    fn save_git_patch_if_needed(&self) -> Result<(), AgentError> {
        if let Some(patch_path_str) = &self.base_agent.patch_path {
            if let Some(project_path_str) = &self.base_agent.project_path {
                info!(
                    "Attempting to save git diff to patch_path: {}",
                    patch_path_str
                );
                match crate::utils::git_utils::get_git_diff(
                    project_path_str,
                    self.base_agent.base_commit.as_deref(),
                ) {
                    Ok(diff_content) => {
                        match std::fs::write(patch_path_str, diff_content) {
                            Ok(_) => {
                                info!("Successfully saved git diff to {}", patch_path_str);
                            }
                            Err(e) => {
                                error!("Failed to write patch file to {}: {}", patch_path_str, e);
                                // Decide if this should be a critical error for the agent's task
                                // For now, just log it, as the main task might have succeeded.
                                // Could also return an error:
                                // return Err(AgentError::ToolError(crate::tools::ToolError::FileWriteError(format!("Failed to write patch to {}: {}", patch_path_str, e))));
                            }
                        }
                    }
                    Err(e) => {
                        error!(
                            "Failed to get git diff for saving to patch_path {}: {}",
                            patch_path_str, e
                        );
                        // return Err(AgentError::ToolError(crate::tools::ToolError::InternalError(format!("Failed to get git diff: {}", e))));
                    }
                }
            } else {
                warn!(
                    "patch_path ({}) is set, but project_path is not. Cannot save patch.",
                    patch_path_str
                );
            }
        }
        Ok(())
    }
}

#[async_trait]
impl Agent for TraeAgent {
    /// Returns the name of this agent implementation.
    fn get_name(&self) -> String {
        "TraeAgentRs".to_string()
    }

    /// Initializes a new task for the `TraeAgent`.
    ///
    /// Sets up the initial conversation history with a system prompt and the user's task,
    /// and processes task-specific arguments like `project_path` and `must_patch`.
    async fn new_task(&mut self, task: String, task_args: Option<Value>) -> Result<(), AgentError> {
        info!(agent_name = %self.get_name(), task = %task, "Received new task");
        self.base_agent.current_task = Some(task.clone());
        self.base_agent.conversation_history.clear();

        let mut recorder_extra_args = HashMap::new();

        if let Some(args) = &task_args { // Borrow args
            if let Some(path_val) = args.get("project_path") {
                if let Some(path_str) = path_val.as_str() {
                    self.base_agent.project_path = Some(path_str.to_string());
                    recorder_extra_args.insert("project_path".to_string(), path_str.to_string());
                    debug!("Set project path to: {}", path_str);
                }
            }
            if let Some(patch_val) = args.get("must_patch") {
                if let Some(patch_bool) = patch_val.as_bool() {
                    self.base_agent.must_patch = patch_bool;
                    debug!("Set must_patch to: {}", patch_bool);
                } else if let Some(patch_str) = patch_val.as_str() {
                    self.base_agent.must_patch = patch_str.eq_ignore_ascii_case("true");
                    debug!(
                        "Set must_patch (from string) to: {}",
                        self.base_agent.must_patch
                    );
                }
            }
            if let Some(bc_val) = args.get("base_commit") {
                if let Some(bc_str) = bc_val.as_str() {
                    self.base_agent.base_commit = Some(bc_str.to_string());
                    recorder_extra_args.insert("base_commit".to_string(), bc_str.to_string());
                    debug!("Set base_commit to: {}", bc_str);
                }
            }
            if let Some(pp_val) = args.get("patch_path") {
                if let Some(pp_str) = pp_val.as_str() {
                    self.base_agent.patch_path = Some(pp_str.to_string());
                    // Not typically needed for trajectory_extra_args, but can be added if desired
                    debug!("Set patch_path to: {}", pp_str);
                }
            }
            // Add other relevant args to recorder_extra_args if needed
            recorder_extra_args.insert("must_patch".to_string(), self.base_agent.must_patch.to_string());
        }

        // Start trajectory recording if recorder is available
        if let Some(recorder) = self.base_agent.trajectory_recorder.as_mut() {
            let _ = recorder.start_recording(
                task.clone(),
                self.base_agent.llm_client.get_provider_name(), // Assuming LLMClient has such a method
                self.base_agent.config.get_current_provider_config().map_or("unknown_model".to_string(), |pc| pc.model.clone()), // Get model from config
                self.base_agent.max_steps,
                Some(recorder_extra_args).filter(|m| !m.is_empty()), // Only pass if not empty
            );
        }

        self.base_agent.conversation_history.push(LLMMessage {
            role: MessageRole::System,
            content: Some(self.get_system_prompt()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });

        let mut user_message_content = String::new();
        let problem_statement_text = if let Some(args_val) = &task_args {
            if let Some(issue_val) = args_val.get("issue") {
                issue_val.as_str().map(|s| s.to_string())
            } else {
                None
            }
        } else {
            None
        };

        if let Some(issue_text) = problem_statement_text {
            user_message_content.push_str(&format!(
                "[Problem statement]: We're currently solving the following issue within our repository. Here's the issue text:\n{}\n",
                issue_text
            ));
        } else {
            // Fallback to using the main task string if 'issue' is not provided
            user_message_content.push_str(&format!("[Problem statement]: {}\n", task));
        }

        if let Some(project_path) = &self.base_agent.project_path {
            user_message_content.push_str(&format!("\n[Project root path]: {}\n", project_path));
        }
        // Ensure there's a blank line if both problem statement and project path are present.
        // The format! macro for project_path already adds a newline at the start if user_message_content is not empty.


        self.base_agent.conversation_history.push(LLMMessage {
            role: MessageRole::User,
            content: Some(user_message_content.trim_end().to_string() + "\n"), // Ensure single trailing newline
            name: None,
            tool_calls: None,
            tool_call_id: None,
        });

        Ok(())
    }

    /// Executes the current task for the `TraeAgent`.
    ///
    /// This method orchestrates the agent's interaction with the LLM and tools
    /// by calling the `common_execute_task_loop`. It provides `TraeAgent`-specific
    /// logic for determining when a task is complete (including patch validation)
    /// via closures passed to the loop.
    async fn execute_task(
        &mut self,
        event_sender: Option<mpsc::Sender<AgentEvent>>,
    ) -> Result<super::base_agent::AgentExecution, AgentError> {
        if self.base_agent.current_task.is_none() {
            return Err(AgentError::TaskSetupFailed(
                "No task has been set. Call new_task first.".to_string(),
            ));
        }
        if self.base_agent.conversation_history.len() < 2 {
            return Err(AgentError::TaskSetupFailed(
                "Initial prompts not set up correctly.".to_string(),
            ));
        }

        let initial_messages_clone = self.base_agent.conversation_history.clone();

        // Capture necessary values from self.base_agent *before* the mutable borrow for common_execute_task_loop
        let must_patch_val = self.base_agent.must_patch;
        // Clone project_path to avoid lifetime issues if it's needed beyond the borrow of self.base_agent
        // Or ensure fn_should_stop can handle Option<&str> if its lifetime is tied to self.base_agent correctly.
        // For simplicity here, let's clone it if it exists.
        let project_path_cloned_opt: Option<String> = self.base_agent.project_path.clone();
        let base_commit_cloned_opt: Option<String> = self.base_agent.base_commit.clone();

        let execution_result = common_execute_task_loop(
            &mut self.base_agent,
            initial_messages_clone,
            event_sender,
            // Pass closures that call the static methods, using captured values
            &|llm_response, step, max_steps| {
                TraeAgent::fn_should_stop(
                    llm_response,
                    step,
                    max_steps,
                    must_patch_val,
                    project_path_cloned_opt.as_deref(),
                    base_commit_cloned_opt.as_deref(), // Pass captured base_commit
                )
            },
            &|llm_response| TraeAgent::fn_process_llm_response_for_completion(llm_response),
        )
        .await;

        // After task execution, try to save the patch if configured
        if let Err(e) = self.save_git_patch_if_needed() {
            // Log the error, but don't necessarily make the whole task fail due to patch saving error.
            // The main execution result is more important.
            warn!("Failed to save git patch (if configured): {:?}", e);
        }

        execution_result
    }

    async fn execute_interactive_turn(
        &mut self,
        event_sender: Option<mpsc::Sender<AgentEvent>>,
    ) -> Result<Vec<LLMMessage>, AgentError> {
        if self.base_agent.current_task.is_none() {
            return Err(AgentError::TaskSetupFailed(
                "No task has been set for interactive turn. Call new_task first.".to_string(),
            ));
        }
        // `new_task` should have already been called by the CLI,
        // setting up current_task and initial messages (system + user query).
        // The conversation_history in base_agent is the source of truth.

        let initial_messages_count = self.base_agent.conversation_history.len();
        if initial_messages_count < 1 {
            // Should be at least 1 (system) or 2 (system + user)
            return Err(AgentError::TaskSetupFailed(
                "Interactive turn started with empty history.".to_string(),
            ));
        }

        // For an interactive turn, we expect a limited number of "steps" or cycles.
        // Typically:
        // 1. LLM responds to user. If no tools, turn ends.
        // 2. If LLM calls tools, tools execute, then LLM responds again. Turn ends.
        // So, max_steps for common_execute_task_loop should be small, e.g., 2.
        // We'll use the agent's configured max_steps from InteractiveArgs, but cap it for a single turn.
        let max_steps_for_this_turn = std::cmp::min(self.base_agent.max_steps, 2); // Cap at 2 "micro-steps"

        // Store a copy of messages before this turn's execution, to identify newly added messages.
        // This is tricky because common_execute_task_loop modifies history in place.
        // We need to return only the messages generated *during this call*.

        // The `common_execute_task_loop` will append to `self.base_agent.conversation_history`.
        // We need to capture what was added.
        let history_before_execution = self.base_agent.conversation_history.clone();

        let execution_result = common_execute_task_loop(
            &mut self.base_agent,
            history_before_execution.clone(), // Pass the current full history
            event_sender,
            &|llm_response, step, _max_steps_from_common_loop| {
                // _max_steps_from_common_loop is ignored here
                // We use max_steps_for_this_turn for interactive logic
                TraeAgent::fn_should_stop_interactive(llm_response, step, max_steps_for_this_turn)
            },
            &|llm_response| TraeAgent::fn_extract_assistant_response_interactive(llm_response),
        )
        .await;

        // After common_execute_task_loop, self.base_agent.conversation_history contains all messages
        // up to the end of this turn. We need to extract only those added during this call.
        let history_after_execution = &self.base_agent.conversation_history;
        let new_messages_this_turn =
            if history_after_execution.len() > history_before_execution.len() {
                history_after_execution[history_before_execution.len()..].to_vec()
            } else {
                // This case (history shrinking or same size but loop ran) might indicate an issue
                // or simply that the loop exited without adding new messages (e.g. immediate error).
                // If execution_result is an error, that's the primary problem.
                // If Ok, but no new messages, it's unusual but possible if loop exited on first check.
                Vec::new()
            };

        match execution_result {
            Ok(_agent_execution_summary) => {
                // The actual "result" of the turn for interactive mode is the set of new messages.
                // The AgentExecution summary might be useful for logging but not directly returned here.
                Ok(new_messages_this_turn)
            }
            Err(e) => {
                error!("Error during interactive turn execution: {:?}", e);
                // If an error occurred, new_messages_this_turn might be empty or partial.
                // The error itself is more important.
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::llm::base_client::ModelParameters;
    use crate::tools::BashTool;
    use std::collections::HashMap;

    fn create_test_config() -> Arc<Config> {
        let mut providers = HashMap::new();
        providers.insert(
            "openai".to_string(),
            ModelParameters {
                api_key: Some("test_key".to_string()),
                model: "gpt-test".to_string(),
                max_tokens: Some(100),
                temperature: 0.5,
                top_p: 1.0,
                top_k: None,
                parallel_tool_calls: true,
                max_retries: crate::config::default_max_retries(),
                base_url: None,
                api_version: None,
                candidate_count: None,
                stop_sequences: None,
            },
        );
        Arc::new(Config {
            default_provider: "openai".to_string(),
            max_steps: 5,
            model_providers: providers,
            lakeview_config: None,  // Added
            enable_lakeview: false, // Added (or true, depending on test needs)
            working_dir: Some("/tmp".to_string()),
        })
    }

    fn create_test_tool_registry() -> Arc<ToolRegistry> {
        let mut registry = ToolRegistry::new();
        registry.register(BashTool::new());
        Arc::new(registry)
    }

    #[tokio::test]
    async fn test_trae_agent_new_task() {
        let config = create_test_config();
        let tool_registry = create_test_tool_registry();
        let mut agent = TraeAgent::try_new(config, tool_registry, None) // Pass None for trajectory_file_path
            .await
            .expect("Failed to create agent");

        let task_desc = "Test task: create a file.".to_string();
        let mut task_args_map = serde_json::Map::new();
        task_args_map.insert(
            "project_path".to_string(),
            Value::String("/test/path".to_string()),
        );
        task_args_map.insert("must_patch".to_string(), Value::Bool(true));
        let task_args_val = Value::Object(task_args_map);

        agent
            .new_task(task_desc.clone(), Some(task_args_val))
            .await
            .unwrap();

        assert_eq!(agent.base_agent.current_task, Some(task_desc.clone())); // Clone task_desc for this comparison
        assert_eq!(
            agent.base_agent.project_path,
            Some("/test/path".to_string())
        );
        assert!(agent.base_agent.must_patch); // Corrected: use assert! for boolean
        assert_eq!(agent.base_agent.conversation_history.len(), 2);
        assert_eq!(
            agent.base_agent.conversation_history[0].role,
            MessageRole::System
        );
        assert!(agent.base_agent.conversation_history[1]
            .content
            .as_deref()
            .unwrap()
            .contains(&task_desc));
        assert!(agent.base_agent.conversation_history[1]
            .content
            .as_deref()
            .unwrap()
            .contains("[Project root path]: /test/path"));
    }

    // Tests for fn_should_stop
    mod test_fn_should_stop {
        use super::*; // To get TraeAgent and its methods, LLMResponse etc.
        use crate::agent::base_agent::StopReason;
        // Corrected imports based on llm/base_client.rs
        use crate::llm::base_client::{LLMMessage, LLMResponse, LLMResponseChoice, ToolCall, ToolCallFunction};

        fn mock_llm_response(content: Option<String>, tool_calls: Option<Vec<ToolCall>>) -> LLMResponse {
            LLMResponse {
                id: "test_response_id".to_string(),
                object: "chat.completion".to_string(), // Added field
                created: 0, // Added field, default to 0 for mock
                model: "test_model".to_string(),
                choices: vec![LLMResponseChoice { // Corrected type
                    index: 0,
                    message: LLMMessage {
                        role: MessageRole::Assistant,
                        content,
                        name: None,
                        tool_calls, // Uses corrected ToolCall type
                        tool_call_id: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
                usage: None,
                // system_fingerprint: None, // Removed field
            }
        }

        fn task_done_tool_call() -> ToolCall { // Corrected type
            ToolCall { // Corrected type
                id: "call_task_done_123".to_string(),
                tool_type: "function".to_string(), // field name is tool_type
                function: ToolCallFunction { // Corrected type
                    name: "task_done".to_string(),
                    arguments: "{}".to_string(),
                },
            }
        }

        #[test]
        fn test_stop_max_steps_reached() {
            let response = mock_llm_response(Some("hello".to_string()), None);
            let reason = TraeAgent::fn_should_stop(&response, 5, 5, false, None, None);
            assert_eq!(reason, StopReason::MaxStepsReached);
        }

        #[test]
        fn test_stop_by_task_done_tool_no_patch_required() {
            let response = mock_llm_response(None, Some(vec![task_done_tool_call()]));
            let reason = TraeAgent::fn_should_stop(&response, 1, 5, false, None, None);
            assert_eq!(reason, StopReason::TaskCompleted);
        }

        #[test]
        fn test_stop_by_textual_completion_no_patch_required() {
            let response = mock_llm_response(Some("The task completed successfully.".to_string()), None);
            let reason = TraeAgent::fn_should_stop(&response, 1, 5, false, None, None);
            assert_eq!(reason, StopReason::TaskCompleted);
        }

        #[test]
        fn test_continue_if_no_completion_signal() {
            let response = mock_llm_response(Some("Working on it.".to_string()), None);
            let reason = TraeAgent::fn_should_stop(&response, 1, 5, false, None, None);
            assert_eq!(reason, StopReason::Continue);
        }

        // More complex cases involving must_patch = true would require mocking git_utils.
        // For now, these cover the non-patch-dependent parts of the logic.

        #[test]
        fn test_stop_by_task_done_must_patch_no_project_path() {
            let response = mock_llm_response(None, Some(vec![task_done_tool_call()]));
            let reason = TraeAgent::fn_should_stop(&response, 1, 5, true, None, None); // must_patch = true, no project_path
            assert_eq!(
                reason,
                StopReason::ValidationFailed(
                    "ERROR! 'must_patch' is true, but project_path is not configured for diffing."
                        .to_string()
                )
            );
        }

        #[test]
        fn test_stop_by_textual_completion_must_patch_no_project_path() {
            let response = mock_llm_response(Some("Done.".to_string()), None);
            let reason = TraeAgent::fn_should_stop(&response, 1, 5, true, None, None); // must_patch = true, no project_path
             assert_eq!(
                reason,
                StopReason::ValidationFailed(
                    "ERROR! 'must_patch' is true, but project_path is not configured for diffing."
                        .to_string()
                )
            );
        }

        // To test patch validation success/failure, we'd need to mock:
        // - crate::utils::git_utils::get_git_diff
        // - crate::utils::git_utils::remove_patches_to_tests
        // This is non-trivial for unit tests without a mocking framework or feature flags.
        // We are testing the logic paths that lead *to* those calls here.
    }

    // TODO: Add tests for TraeAgent::new_task prompt formatting with 'issue'
    #[cfg(test)]
    mod test_new_task_prompt {
        use super::*;
        use serde_json::json;

        #[tokio::test]
        async fn test_new_task_with_issue_arg() {
            let config = create_test_config(); // Assuming create_test_config is in outer scope
            let tool_registry = create_test_tool_registry(); // Assuming create_test_tool_registry is in outer scope
            let mut agent = TraeAgent::try_new(config, tool_registry, None)
                .await
                .unwrap();

            let task_desc = "Overall task title".to_string();
            let issue_text = "Specific bug description here.".to_string();
            let project_path_text = "/test/project".to_string();

            let task_args_val = json!({
                "project_path": project_path_text,
                "issue": issue_text
            });

            agent.new_task(task_desc.clone(), Some(task_args_val)).await.unwrap();

            let user_message = agent.base_agent.conversation_history.iter().find(|m| m.role == MessageRole::User).unwrap();
            let expected_problem_statement = format!("[Problem statement]: We're currently solving the following issue within our repository. Here's the issue text:\n{}\n", issue_text);
            let expected_project_path_statement = format!("\n[Project root path]: {}\n", project_path_text);

            assert!(user_message.content.as_ref().unwrap().contains(&expected_problem_statement));
            assert!(user_message.content.as_ref().unwrap().contains(&expected_project_path_statement));
            assert!(!user_message.content.as_ref().unwrap().contains(&task_desc)); // Task desc shouldn't be in problem statement if issue is used
        }

        #[tokio::test]
        async fn test_new_task_without_issue_arg_fallback_to_task_desc() {
            let config = create_test_config();
            let tool_registry = create_test_tool_registry();
            let mut agent = TraeAgent::try_new(config, tool_registry, None)
                .await
                .unwrap();

            let task_desc = "Fix the login button".to_string();
            let project_path_text = "/test/another/project".to_string();

            let task_args_val = json!({
                "project_path": project_path_text
            }); // No "issue" field

            agent.new_task(task_desc.clone(), Some(task_args_val)).await.unwrap();

            let user_message = agent.base_agent.conversation_history.iter().find(|m| m.role == MessageRole::User).unwrap();
            let expected_problem_statement = format!("[Problem statement]: {}\n", task_desc); // Fallback
            let expected_project_path_statement = format!("\n[Project root path]: {}\n", project_path_text);

            assert!(user_message.content.as_ref().unwrap().contains(&expected_problem_statement));
            assert!(user_message.content.as_ref().unwrap().contains(&expected_project_path_statement));
        }
    }

    #[cfg(test)]
    mod test_patch_saving {
        use super::*;
        use serde_json::json;
        use tempfile::tempdir;
        use std::fs;
        use std::path::Path;
        use std::process::Command;

        // Helper to init a git repo and make commits - adapted from git_utils tests
        fn setup_initial_repo_for_diff(dir: &Path) -> Result<String, anyhow::Error> {
            Command::new("git").arg("init").current_dir(dir).status()?;
            Command::new("git").args(["config", "user.name", "Test User"]).current_dir(dir).status()?;
            Command::new("git").args(["config", "user.email", "test@example.com"]).current_dir(dir).status()?;

            fs::write(dir.join("file.txt"), "initial content")?;
            Command::new("git").arg("add").arg("file.txt").current_dir(dir).status()?;
            Command::new("git").arg("commit").arg("-m").arg("Initial commit").current_dir(dir).status()?;
            let base_commit_hash = String::from_utf8(
                Command::new("git").arg("rev-parse").arg("HEAD").current_dir(dir).output()?.stdout
            )?.trim().to_string();

            // Make the change that we want to see in the patch
            fs::write(dir.join("file.txt"), "modified content")?;
            // Commit this change so HEAD moves past base_commit
            Command::new("git").arg("add").arg("file.txt").current_dir(dir).status()?;
            Command::new("git").arg("commit").arg("-m").arg("Modified file.txt").current_dir(dir).status()?;

            Ok(base_commit_hash) // Return the hash *before* the modification
        }

        #[tokio::test]
        async fn test_patch_saving_successful() {
            let project_dir = tempdir().unwrap();
            let patch_dir = tempdir().unwrap();
            let base_commit = setup_initial_repo_for_diff(project_dir.path()).unwrap();

            let config = create_test_config();
            let tool_registry = create_test_tool_registry();
            let mut agent = TraeAgent::try_new(config, tool_registry, None).await.unwrap();

            let patch_file_path = patch_dir.path().join("test.patch");
            agent.base_agent.project_path = Some(project_dir.path().to_str().unwrap().to_string());
            agent.base_agent.base_commit = Some(base_commit);
            agent.base_agent.patch_path = Some(patch_file_path.to_str().unwrap().to_string());

            agent.save_git_patch_if_needed().unwrap();

            assert!(patch_file_path.exists());
            let patch_content = fs::read_to_string(patch_file_path).unwrap();
            assert!(patch_content.contains("--- a/file.txt"));
            assert!(patch_content.contains("+++ b/file.txt"));
            assert!(patch_content.contains("-initial content"));
            assert!(patch_content.contains("+modified content"));
        }

        #[tokio::test]
        async fn test_patch_saving_no_patch_path() {
            let project_dir = tempdir().unwrap();
            setup_initial_repo_for_diff(project_dir.path()).unwrap(); // Setup repo but don't need base_commit

            let config = create_test_config();
            let tool_registry = create_test_tool_registry();
            let mut agent = TraeAgent::try_new(config, tool_registry, None).await.unwrap();

            agent.base_agent.project_path = Some(project_dir.path().to_str().unwrap().to_string());
            agent.base_agent.patch_path = None; // No patch path set

            // This should do nothing and not panic
            agent.save_git_patch_if_needed().unwrap();
            // Assert no file was created (difficult to check universally, rely on no panic and code logic)
        }

        #[tokio::test]
        async fn test_patch_saving_no_project_path() {
            let patch_dir = tempdir().unwrap();
            let patch_file_path = patch_dir.path().join("test.patch");

            let config = create_test_config();
            let tool_registry = create_test_tool_registry();
            let mut agent = TraeAgent::try_new(config, tool_registry, None).await.unwrap();

            agent.base_agent.project_path = None; // No project path
            agent.base_agent.patch_path = Some(patch_file_path.to_str().unwrap().to_string());

            // This should log a warning and not panic
            agent.save_git_patch_if_needed().unwrap();
            assert!(!patch_file_path.exists()); // File should not be created
        }
         #[tokio::test]
        async fn test_new_task_parses_patch_path() {
            let config = create_test_config();
            let tool_registry = create_test_tool_registry();
            let mut agent = TraeAgent::try_new(config, tool_registry, None).await.unwrap();

            let task_args = json!({
                "project_path": "/tmp/dummy",
                "patch_path": "/tmp/output.patch"
            });
            agent.new_task("test".to_string(), Some(task_args)).await.unwrap();
            assert_eq!(agent.base_agent.patch_path, Some("/tmp/output.patch".to_string()));
        }
    }
}
