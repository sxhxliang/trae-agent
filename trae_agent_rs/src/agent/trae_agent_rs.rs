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
    /// A tuple `(bool, Option<String>)`:
    /// * The boolean is `true` if the agent should stop, `false` otherwise.
    /// * The `Option<String>` contains an error message to be sent to the LLM if stopping
    ///   is premature due to a validation failure (e.g., empty patch when required).
    fn fn_should_stop(
        llm_response: &LLMResponse,
        current_step_number: u32,
        max_steps: u32,
        must_patch: bool,
        project_path: Option<&str>,
        base_commit: Option<&str>, // Added base_commit parameter
    ) -> (bool, Option<String>) {
        if current_step_number >= max_steps {
            warn!("Max steps reached, forcing stop.");
            return (true, None);
        }
        if let Some(tool_calls) = &llm_response.choices[0].message.tool_calls {
            for tc in tool_calls {
                if tc.function.name == "task_done" {
                    info!("'task_done' tool called by LLM.");
                    if must_patch {
                        if let Some(proj_p) = project_path {
                            match crate::utils::git_utils::get_git_diff(proj_p, base_commit) {
                                // Use base_commit
                                Ok(model_patch) => {
                                    let patch = crate::utils::git_utils::remove_patches_to_tests(
                                        &model_patch,
                                    );
                                    if patch.trim().is_empty() {
                                        warn!("Task completion signaled with 'task_done', but 'must_patch' is true and generated patch is empty (after filtering tests). Task not considered done.");
                                        return (false, Some("ERROR! Your Patch is empty. Please provide a patch that fixes the problem.".to_string()));
                                    }
                                    info!("Patch validation successful for 'task_done'.");
                                    return (true, None); // Task done and patch is valid
                                }
                                Err(e) => {
                                    error!("Failed to get git diff for patch validation: {}", e);
                                    // Treat as patch validation failure, ask LLM to retry or fix
                                    return (false, Some(format!("ERROR! Could not verify patch due to git diff error: {}. Please try the fix again.", e)));
                                }
                            }
                        } else {
                            warn!("'must_patch' is true, but no project_path is available for git diff. Assuming task not complete.");
                            return (false, Some("ERROR! 'must_patch' is true, but project_path is not configured for diffing.".to_string()));
                        }
                    } else {
                        // must_patch is false, so task_done is enough
                        return (true, None);
                    }
                }
            }
        }
        (false, None) // Default: don't stop, no error message
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
    ) -> (bool, Option<String>) {
        if current_step_in_turn >= max_steps_for_turn {
            return (true, None); // Reached max micro-steps for this turn
        }

        let message = &llm_response.choices[0].message;
        if message.tool_calls.is_none()
            || message.tool_calls.as_ref().map_or(true, |tc| tc.is_empty())
        {
            // If there are no tool calls, this is a direct response to the user, so stop.
            return (true, None);
        }

        // If there ARE tool calls, we don't stop yet. We want to execute them and get one more LLM response.
        // The max_steps_for_turn (e.g., 2) will handle stopping after tools + next LLM response.
        // If current_step_in_turn is 1 and there are tool calls, we'll proceed to step 2.
        // If current_step_in_turn is 2 (meaning tools were called in step 1, and this is LLM response after tools), we stop.
        if current_step_in_turn >= 1
            && (message.tool_calls.is_some() && !message.tool_calls.as_ref().unwrap().is_empty())
        {
            // This condition means: if we are past the first step AND there are tool calls,
            // we let max_steps_for_turn (e.g. 2) handle it.
            // If on step 1, and there are tool calls, we continue.
            // If on step 2 (meaning step 1 had tool calls), we stop regardless of this response.
            // This logic is simplified by simply checking current_step_in_turn >= max_steps_for_turn above.
        }

        // If it's the first step in the turn (current_step_in_turn == 1) and there are tool calls,
        // we want to continue to execute tools and get the next LLM response.
        // If it's a later step (e.g. current_step_in_turn == 2, meaning tools were just executed),
        // then we should stop with this LLM response.
        if current_step_in_turn > 1 && message.tool_calls.is_some() {
            // This LLM response came after tool execution. Stop now.
            return (true, None);
        }

        (false, None) // Default: continue if within max_steps_for_turn and tools were just called on step 1.
    }

    /// Processes the LLM's response for an interactive turn to get the assistant's message.
    fn fn_extract_assistant_response_interactive(llm_response: &LLMResponse) -> Option<String> {
        // For interactive mode, the "final result" is simply the assistant's content.
        llm_response.choices[0].message.content.clone()
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

        let mut user_message_content = format!("[Problem statement]: {}\n", task);
        if let Some(project_path) = &self.base_agent.project_path {
            user_message_content.push_str(&format!("[Project root path]: {}\n", project_path));
        }

        self.base_agent.conversation_history.push(LLMMessage {
            role: MessageRole::User,
            content: Some(user_message_content),
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

        common_execute_task_loop(
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
        .await
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
}
