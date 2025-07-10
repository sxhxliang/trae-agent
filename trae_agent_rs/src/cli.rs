//! # CLI Module
//!
//! Handles command-line argument parsing and dispatching to appropriate handlers
//! for the Trae Rust Agent. It uses the `clap` crate for parsing.

use crate::config::Config;
use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run a task using Trae Agent
    Run(RunArgs),
    /// Start an interactive session with Trae Agent
    Interactive(InteractiveArgs),
    /// Show current configuration settings
    ShowConfig(ShowConfigArgs),
    /// Show available tools and their descriptions
    Tools(ToolsArgs),
}

#[derive(Parser, Debug)]
pub struct RunArgs {
    #[arg(index = 1)]
    pub task: String,
    #[arg(short, long)]
    pub provider: Option<String>,
    #[arg(short, long)]
    pub model: Option<String>,
    #[arg(short, long)]
    pub api_key: Option<String>,
    #[arg(long)]
    pub max_steps: Option<u32>,
    #[arg(short, long)]
    pub working_dir: Option<String>,
    #[arg(long, short = 'M', alias = "must-patch")]
    pub must_patch: bool,
    #[arg(long, default_value = "trae_config.json")]
    pub config_file: String,
    #[arg(short, long)]
    pub trajectory_file: Option<String>,
    #[arg(long, short = 'P', alias = "patch-path")]
    pub patch_path: Option<String>,
    #[arg(long, alias = "base-commit")]
    pub base_commit: Option<String>,
}

#[derive(Parser, Debug)]
pub struct InteractiveArgs {
    #[arg(short, long)]
    pub provider: Option<String>,
    #[arg(short, long)]
    pub model: Option<String>,
    #[arg(short, long)]
    pub api_key: Option<String>,
    #[arg(long, default_value = "trae_config.json")]
    pub config_file: String,
    #[arg(long, default_value_t = 20)]
    pub max_steps: u32,
    #[arg(short, long)]
    pub trajectory_file: Option<String>,
}

#[derive(Parser, Debug)]
pub struct ShowConfigArgs {
    #[arg(long, default_value = "trae_config.json")]
    pub config_file: String,
}

#[derive(Parser, Debug)]
pub struct ToolsArgs {} // No arguments needed for listing tools

use crate::agent::base_agent::AgentEvent; // Removed AgentStep, AgentExecution
use crate::agent::{Agent, TraeAgent};
use crate::llm::base_client::LLMMessage;
use crate::llm::LLMClient; // Restored LLMClient for Lakeview type annotations
use crate::llm::MessageRole; // Added import for MessageRole
                             // OpenAIClient is used by TraeAgent internally, not directly needed here for handle_interactive
                             // LLMClient is used by TraeAgent internally
                             // Tool specific imports (BashTool, EditTool etc.) are not needed as ToolRegistry handles them.
use crate::tools::ToolRegistry;

// Removed: mod cli_tools_handler;

use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

pub async fn handle_run(args: RunArgs) -> anyhow::Result<()> {
    info!("Starting 'run' command with task: {}", args.task);

    let config = match Config::load(
        &args.config_file,
        args.provider.clone(),
        args.model.clone(),
        args.api_key.clone(),
        args.max_steps,
        args.working_dir.clone(),
    ) {
        Ok(cfg) => Arc::new(cfg),
        Err(e) => {
            error!("Failed to load configuration: {:?}", e);
            return Err(e);
        }
    };
    info!(
        "Configuration loaded successfully. Default provider: {}",
        config.default_provider
    );

    let tool_registry = Arc::new(ToolRegistry::default());
    info!(
        "ToolRegistry initialized with {} tools.",
        tool_registry.get_all_tools_arc().len()
    );

    let trajectory_path_buf = args.trajectory_file.map(PathBuf::from);

    let mut agent = match TraeAgent::try_new(config.clone(), tool_registry.clone(), trajectory_path_buf).await {
        Ok(ag) => ag,
        Err(e) => {
            error!("Failed to create TraeAgent: {:?}", e);
            return Err(anyhow::anyhow!("Agent creation failed: {}", e));
        }
    };
    info!("TraeAgent created successfully: {}", agent.get_name());

    let mut task_agent_args = serde_json::Map::new();
    if let Some(wd) = &config.working_dir {
        task_agent_args.insert(
            "project_path".to_string(),
            serde_json::Value::String(wd.clone()),
        );
    }
    task_agent_args.insert(
        "must_patch".to_string(),
        serde_json::Value::Bool(args.must_patch),
    );

    let patch_path_clone_for_task_args = args.patch_path.clone();
    if let Some(pp) = patch_path_clone_for_task_args {
        task_agent_args.insert("patch_path".to_string(), serde_json::Value::String(pp));
    }
    if let Some(bc) = args.base_commit.clone() {
        task_agent_args.insert("base_commit".to_string(), serde_json::Value::String(bc));
    }

    if let Err(e) = agent
        .new_task(
            args.task.clone(),
            Some(serde_json::Value::Object(task_agent_args)),
        )
        .await
    {
        error!("Failed to setup new task for agent: {:?}", e);
        return Err(anyhow::anyhow!("Task setup failed: {}", e));
    }
    info!("New task '{}' initialized for agent.", args.task);

    let (event_tx, mut event_rx) = mpsc::channel(100);

    let console_updater_task = tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            match event {
                AgentEvent::StepBegin(step_num) => {
                    println!("\n[AGENT EVENT] Step {} Starting...", step_num);
                }
                AgentEvent::StepStateChange(_step_num, new_state) => {
                    println!("[AGENT EVENT] State: {:?}", new_state);
                }
                AgentEvent::LLMRequestSent(_step_num, messages) => {
                    if let Some(last_msg) = messages.last() {
                        let content_preview = last_msg.content.as_deref().unwrap_or_default();
                        println!(
                            "[AGENT EVENT] Sending to LLM (last msg role: {:?}): {:.100}...",
                            last_msg.role, content_preview
                        );
                    }
                }
                AgentEvent::LLMResponseReceived(_step_num, response) => {
                    if let Some(choice) = response.choices.first() {
                        let content_preview = choice.message.content.as_deref().unwrap_or_default();
                        println!("[AGENT EVENT] LLM Response: {:.100}...", content_preview);
                        if let Some(tool_calls) = &choice.message.tool_calls {
                            if !tool_calls.is_empty() {
                                println!(
                                    "[AGENT EVENT] LLM requested {} tool call(s).",
                                    tool_calls.len()
                                );
                            }
                        }
                    }
                }
                AgentEvent::ToolCallAttempt(_step_num, tool_call) => {
                    println!(
                        "[AGENT EVENT] Tool call: {} with args {:.100}...",
                        tool_call.function.name, tool_call.function.arguments
                    );
                }
                AgentEvent::ToolCallResult(_step_num, tool_result) => {
                    let result_preview = tool_result
                        .result
                        .as_deref()
                        .or(tool_result.error.as_deref())
                        .unwrap_or_default();
                    println!(
                        "[AGENT EVENT] Tool result (ID: {}): Success: {}, Details: {:.100}...",
                        tool_result.tool_call_id, tool_result.success, result_preview
                    );
                }
                AgentEvent::TaskCompleted(_) | AgentEvent::TaskFailed(_) => { /* Final summary handles this */
                }
                AgentEvent::StatusUpdate(msg) => {
                    println!("[AGENT EVENT] Status: {}", msg);
                }
            }
        }
    });

    let execution_result = match agent.execute_task(Some(event_tx)).await {
        Ok(exec_res) => {
            info!("Task execution process finished by agent logic.");
            exec_res
        }
        Err(e) => {
            error!("Task execution failed with an error: {:?}", e);
            // `agent` is dropped here, closing `event_tx`.
            // `console_updater_task` will complete once it processes remaining events.
            // We await it *after* this match block.
            return Err(anyhow::anyhow!("Task execution error: {}", e));
        }
    };

    // Drop agent explicitly here if not already out of scope, to ensure event_tx is dropped.
    // Actually, agent is dropped after the match if Ok, or before return if Err.
    // So event_tx will be dropped, allowing console_updater_task to finish.
    if let Err(e) = console_updater_task.await {
        error!("Console updater task panicked or was cancelled: {:?}", e);
    }

    println!("\n--- Task Execution Summary ---");
    println!("Task: {}", execution_result.task);
    println!("Success: {}", execution_result.success);
    if let Some(end_time) = execution_result.end_time {
        println!(
            "Execution Time: {}s",
            end_time - execution_result.start_time
        );
    } else {
        println!("Execution Time: Not available (task did not set end time)");
    }
    println!("Total Steps: {}", execution_result.steps.len());
    if let Some(tokens) = &execution_result.total_tokens_used { // Borrow tokens
        println!("Total Tokens Used: {:?}", tokens); // Use {:?} for debug printing
    }
    if let Some(ref res) = execution_result.final_result {
        println!("Final Result: {}", res);
    }
    if let Some(ref err_msg) = execution_result.error_message {
        println!("Error Message: {}", err_msg);
    }

    if let Some(patch_p_ref) = args.patch_path.as_ref() {
        if execution_result.success || args.must_patch {
            if let Some(proj_path) = &config.working_dir {
                match crate::utils::git_utils::get_git_diff(proj_path, args.base_commit.as_deref())
                {
                    Ok(diff_content) => {
                        if let Err(e_write) = std::fs::write(patch_p_ref, diff_content) {
                            error!("Failed to write patch file to {}: {}", patch_p_ref, e_write);
                        } else {
                            info!("Patch file saved to {}", patch_p_ref);
                            println!("Patch file saved to: {}", patch_p_ref);
                        }
                    }
                    Err(e_diff) => {
                        error!("Failed to get git diff for patch file: {}", e_diff);
                    }
                }
            } else {
                error!("Cannot save patch file, project working directory not known.");
            }
        }
    }

    if config.enable_lakeview {
        if let Some(lv_config) = &config.lakeview_config {
            info!("Lakeview enabled, attempting to generate summary...");
            if let Some(lv_model_params_ref) = config.model_providers.get(&lv_config.model_provider)
            {
                let mut specific_lv_params = lv_model_params_ref.clone();
                specific_lv_params.model = lv_config.model_name.clone();

                let lakeview_llm_client_result: Result<Option<Arc<dyn LLMClient>>, anyhow::Error> =
                    match lv_config.model_provider.as_str() {
                        "openai" => crate::llm::OpenAIClient::new(
                            specific_lv_params.api_key.clone(),
                            None,
                            specific_lv_params.clone(),
                        )
                        .await
                        .map(|client| Some(Arc::new(client) as Arc<dyn LLMClient>))
                        .map_err(|e| anyhow::anyhow!("Lakeview OpenAI client error: {}", e)),
                        "anthropic" => crate::llm::AnthropicClient::new(
                            specific_lv_params.api_key.clone(),
                            None,
                            specific_lv_params.clone(),
                        )
                        .await
                        .map(|client| Some(Arc::new(client) as Arc<dyn LLMClient>))
                        .map_err(|e| anyhow::anyhow!("Lakeview Anthropic client error: {}", e)),
                        _ => {
                            error!(
                                "Unsupported Lakeview LLM provider: {}",
                                lv_config.model_provider
                            );
                            Ok(None)
                        }
                    };

                match lakeview_llm_client_result {
                    Ok(Some(client)) => {
                        match crate::utils::lakeview::generate_summary(
                            &execution_result,
                            client,
                            &specific_lv_params,
                        )
                        .await
                        {
                            Ok(summary) => {
                                println!("\n--- Lakeview Summary ---");
                                println!("{}", summary);
                            }
                            Err(e) => {
                                error!("Failed to generate Lakeview summary: {:?}", e);
                                println!("\nFailed to generate Lakeview summary: {}", e);
                            }
                        }
                    }
                    Ok(None) => {
                        warn!("Skipping Lakeview summary due to unsupported provider or client creation issue.");
                    }
                    Err(e) => {
                        error!("Failed to create LLM client for Lakeview: {:?}", e);
                        println!("\nFailed to create LLM client for Lakeview summary: {}", e);
                    }
                }
            } else {
                warn!("Lakeview configured for provider '{}', but its parameters are not found in model_providers.", lv_config.model_provider);
            }
        } else {
            info!("Lakeview enabled but no specific lakeview_config found. Skipping summary.");
        }
    }

    // Trajectory is now saved internally by TrajectoryRecorder when finalize_recording is called
    // within the agent's execution loop if a path was provided during agent initialization.
    // This explicit block is no longer needed.
    // if let Some(traj_path_str) = args.trajectory_file { ... }

    Ok(())
}

pub async fn handle_interactive(args: InteractiveArgs) -> anyhow::Result<()> {
    info!("Starting 'interactive' command session.");

    let config = match Config::load(
        // Removed mut
        &args.config_file,
        args.provider.clone(),
        args.model.clone(),
        args.api_key.clone(),
        Some(args.max_steps), // Pass max_steps for interactive mode from args
        None, // Interactive mode doesn't take working_dir directly at start, uses default
    ) {
        Ok(cfg) => cfg,
        Err(e) => {
            error!("Failed to load configuration: {:?}", e);
            return Err(e);
        }
    };
    info!(
        "Initial configuration loaded. Default provider: {}",
        config.default_provider
    );

    // LLM Client is initialized by the Agent itself based on the config.
    // No need to initialize a separate one here if it's not used directly by handle_interactive.

    // --- Tool Registry and Agent Initialization ---
    let tool_registry = Arc::new(ToolRegistry::default());
    info!(
        "ToolRegistry initialized with {} tools.",
        tool_registry.get_all_tools_arc().len()
    );

    // We need an Arc<Config> for the agent
    let agent_config = Arc::new(config.clone()); // Removed mut, config is already Arc-able

    let trajectory_path_buf = args.trajectory_file.map(PathBuf::from);

    let mut agent = TraeAgent::try_new(agent_config.clone(), tool_registry.clone(), trajectory_path_buf).await?;
    info!("TraeAgent created for interactive mode.");

    let mut rl = DefaultEditor::new().expect("Failed to create rustyline editor");
    if PathBuf::from(".trae_history.txt").exists() {
        let _ = rl.load_history(".trae_history.txt");
    }

    let mut conversation_history: Vec<LLMMessage> = Vec::new();

    println!("Trae Interactive Mode. Type 'exit' or 'quit' to leave.");
    println!("Special commands: config, clear_history, load_config <path> (TODO)");

    loop {
        let readline = rl.readline("trae> ");
        match readline {
            Ok(line) => {
                let _ = rl.add_history_entry(line.as_str());
                let user_input = line.trim();

                if user_input.is_empty() {
                    continue;
                }

                if user_input == "exit" || user_input == "quit" {
                    break;
                } else if user_input == "config" {
                    println!("Current configuration:\n{:#?}", agent_config); // Show agent's current config
                    continue;
                } else if user_input == "clear_history" {
                    conversation_history.clear();
                    println!("Conversation history cleared.");
                    // Also clear agent's internal history if it maintains one separately for interactive
                    // For now, assuming TraeAgent's new_task or equivalent will handle this.
                    // We might need a specific agent.clear_conversation_context() method.
                    // For this iteration, clearing `conversation_history` here is the main step.
                    // The agent will get a fresh history on the next proper input.
                    continue;
                } else if user_input.starts_with("load_config ") {
                    // TODO: Implement config reloading
                    // This would involve:
                    // 1. Parsing the path.
                    // 2. Calling Config::load with the new path.
                    // 3. If successful, re-initializing llm_client, agent_config, and agent.
                    // 4. Clearing conversation_history.
                    // 5. Handling errors gracefully.
                    println!(
                        "Config reloading is not yet implemented. Current config remains active."
                    );
                    continue;
                }

                // Add user message to history
                conversation_history.push(LLMMessage {
                    role: MessageRole::User,
                    content: Some(user_input.to_string()),
                    name: None,
                    tool_calls: None,
                    tool_call_id: None,
                });

                // Create a task for the agent for this turn
                // The "task" is just the user's current input.
                // Agent arguments might be relevant for interactive mode, but keeping it simple for now.
                if let Err(e) = agent.new_task(user_input.to_string(), None).await {
                    error!(
                        "Failed to set new task for agent in interactive mode: {:?}",
                        e
                    );
                    println!("Error setting up task. Please try again.");
                    conversation_history.pop(); // Remove the last user message if task setup failed
                    continue;
                }

                // Execute a "turn" or "step" of the agent.
                // This requires a new method on the Agent trait / TraeAgent.
                // For now, let's assume `execute_interactive_turn` which takes current history
                // and returns new messages from the agent for this turn.
                // It should use `agent.messages` which `new_task` would have set up.

                // The existing `execute_task` is designed for full task completion.
                // We need a version that processes one conversational exchange.
                // This might involve a refactor of `common_execute_task_loop` or a new loop.

                // Placeholder for actual agent interaction logic for one turn:
                println!("Sending to agent: '{}'", user_input);
                // >>> This is where the call to agent.execute_interactive_turn() would go <<<
                // >>> It would update `conversation_history` with the agent's response <<<

                // (No event sender for interactive mode for now, could be added for more detailed output)
                match agent.execute_interactive_turn(None).await {
                    Ok(new_messages) => {
                        if new_messages.is_empty() {
                            println!("Agent processed the input but produced no new messages for the conversation.");
                        }
                        for msg in new_messages {
                            if msg.role == MessageRole::Assistant {
                                if let Some(content) = &msg.content {
                                    println!("Agent: {}", content);
                                } else if msg.tool_calls.is_some() {
                                    // In interactive mode, we might not want to show raw tool calls directly,
                                    // but the LLM's subsequent response after tool execution is what matters.
                                    // This part of the code will be reached if the turn ended AFTER tool calls
                                    // but before a final summarising LLM response.
                                    // The `execute_interactive_turn` is designed to give a final textual response.
                                    println!("Agent: (Thinking/Using tools...)");
                                }
                            }
                            // Add all messages from the turn (thoughts, tool use, final response) to history
                            conversation_history.push(msg);
                        }
                    }
                    Err(e) => {
                        error!("Error during agent's interactive turn: {:?}", e);
                        println!("Agent Error: {}", e);
                        // Optionally, remove the last user message from history if the turn failed critically
                        // For now, keep it to see the context of the error.
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("CTRL-C");
                break;
            }
            Err(ReadlineError::Eof) => {
                println!("CTRL-D");
                break;
            }
            Err(err) => {
                println!("Error reading line: {:?}", err);
                break;
            }
        }
    }

    let _ = rl.save_history(".trae_history.txt");
    println!("Exiting interactive session.");
    Ok(())
}

pub async fn handle_show_config(args: ShowConfigArgs) -> anyhow::Result<()> {
    println!("Attempting to load config from: {}", args.config_file);
    let config = Config::load(&args.config_file, None, None, None, None, None)?;

    println!("\n--- Configuration ---");
    println!("Default Provider: {}", config.default_provider);
    println!("Max Steps: {}", config.max_steps);
    if let Some(wd) = &config.working_dir {
        println!("Working Directory: {}", wd);
    } else {
        println!("Working Directory: Not set (will use current directory)");
    }

    println!("\nModel Providers:");
    for (name, provider_config) in &config.model_providers {
        println!("  Provider: {}", name);
        println!("    Model: {}", provider_config.model);
        println!(
            "    API Key: {}",
            provider_config.api_key.as_deref().unwrap_or("Not set")
        );
        if let Some(mt) = provider_config.max_tokens {
            println!("    Max Tokens: {}", mt);
        }
        println!("    Temperature: {}", provider_config.temperature);
        println!("    Top P: {}", provider_config.top_p);
        if let Some(tk) = provider_config.top_k {
            println!("    Top K: {}", tk);
        }
        println!(
            "    Parallel Tool Calls: {}",
            provider_config.parallel_tool_calls
        );
    }
    println!("--- End Configuration ---");
    Ok(())
}

pub async fn handle_tools_command(_args: ToolsArgs) -> anyhow::Result<()> {
    println!("\n--- Available Tools ---");

    let registry = ToolRegistry::default(); // Create a default registry to list tools
    let tools = registry.get_all_tools_arc();

    if tools.is_empty() {
        println!("No tools are currently registered.");
    } else {
        // Simple table-like format
        // Determine max tool name length for alignment
        let max_name_len = tools.iter().map(|t| t.get_name().len()).max().unwrap_or(20);

        println!("{:<width$} | Description", "Tool Name", width = max_name_len);
        println!("{:-<width$}-|----------------------------------", "-", width = max_name_len);

        for tool in tools {
            let name = tool.get_name();
            let description = tool.get_description();
            // Basic wrapping for description if too long (very naive)
            let short_desc = if description.len() > 70 {
                format!("{}...", &description[..67])
            } else {
                description
            };
            println!("{:<width$} | {}", name, short_desc, width = max_name_len);
        }
    }
    println!("--- End Available Tools ---");
    Ok(())
}
