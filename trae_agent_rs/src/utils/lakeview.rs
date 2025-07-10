//! # Lakeview Summarization Utilities
//!
//! This module provides functionality for generating concise summaries of an agent's
//! execution trajectory using an LLM. It's designed to give a high-level overview
//! of the agent's actions and outcomes.

use crate::agent::{AgentError, AgentExecution};
use crate::llm::{
    // Using re-exports from llm/mod.rs
    LLMError,
    LLMMessage,
    LLMModelParameters,
    MessageRole,
};
// Explicitly import problematic types from base_client as suggested by compiler error
// Removed LLMResponse, LLMResponseChoice, LLMUsage as they are unused in this file.
use crate::llm::LLMClient;

use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn}; // Removed error
                           // use serde_json::Value; // Not directly used here now, but LLM response might be Value

const MAX_SUMMARY_TOKENS: u32 = 500; // Restored for test usage
const MAX_LAKEVIEW_RETRIES: u32 = 3;

const EXTRACTOR_PROMPT: &str = r#"
Given the preceding excerpt, your job is to determine "what task is the agent performing in <this_step>".
Output your answer in two granularities: <task>...</task><details>...</details>.
In the <task> tag, the answer should be concise and general. It should omit ANY bug-specific details, and contain at most 10 words.
In the <details> tag, the answer should complement the <task> tag by adding bug-specific details. It should be informative and contain at most 30 words.

Examples:

<task>The agent is writing a reproduction test script.</task><details>The agent is writing "test_bug.py" to reproduce the bug in XXX-Project's create_foo method not comparing sizes correctly.</details>
<task>The agent is examining source code.</task><details>The agent is searching for "function_name" in the code repository, that is related to the "foo.py:function_name" line in the stack trace.</details>
<task>The agent is fixing the reproduction test script.</task><details>The agent is fixing "test_bug.py" that forgets to import the function "foo", causing a NameError.</details>

Now, answer the question "what task is the agent performing in <this_step>".
Again, provide only the answer with no other commentary. The format should be "<task>...</task><details>...</details>".
"#;

const TAGGER_PROMPT: &str = r#"
Given the trajectory, your job is to determine "what task is the agent performing in the current step".
Output your answer by choosing the applicable tags in the below list for the current step.
If it is performing multiple tasks in one step, choose ALL applicable tags, separated by a comma.

<tags>
WRITE_TEST: It writes a test script to reproduce the bug, or modifies a non-working test script to fix problems found in testing.
VERIFY_TEST: It runs the reproduction test script to verify the testing environment is working.
EXAMINE_CODE: It views, searches, or explores the code repository to understand the cause of the bug.
WRITE_FIX: It modifies the source code to fix the identified bug.
VERIFY_FIX: It runs the reproduction test or existing tests to verify the fix indeed solves the bug.
REPORT: It reports to the user that the job is completed or some progress has been made.
THINK: It analyzes the bug through thinking, but does not perform concrete actions right now.
OUTLIER: A major part in this step does not fit into any tag above, such as running a shell command to install dependencies.
</tags>

<examples>
If the agent is opening a file to examine, output <tags>EXAMINE_CODE</tags>.
If the agent is fixing a known problem in the reproduction test script and then running it again, output <tags>WRITE_TEST,VERIFY_TEST</tags>.
If the agent is merely thinking about the root cause of the bug without other actions, output <tags>THINK</tags>.
</examples>

Output only the tags with no other commentary. The format should be <tags>...</tags>
"#;

fn get_known_tags() -> HashMap<&'static str, &'static str> {
    let mut tags = HashMap::new();
    tags.insert("WRITE_TEST", "☑️");
    tags.insert("VERIFY_TEST", "✅");
    tags.insert("EXAMINE_CODE", "👁️");
    tags.insert("WRITE_FIX", "📝");
    tags.insert("VERIFY_FIX", "🔥");
    tags.insert("REPORT", "📣");
    tags.insert("THINK", "🧠");
    tags.insert("OUTLIER", "⁉️");
    tags
}

#[derive(Debug, Clone)]
pub struct LakeViewStepInfo {
    pub step_number: u32,
    pub desc_task: String,
    pub desc_details: String,
    pub tags_emoji: String,
}

// Helper to format an AgentStep into a string representation for Lakeview LLM calls
fn format_agent_step_for_lakeview(
    agent_step: &crate::agent::base_agent::AgentStep,
) -> Option<String> {
    agent_step.llm_response.as_ref().and_then(|resp| {
        resp.choices.first().and_then(|choice| {
            choice.message.content.as_ref().map(|s| {
                let mut step_summary = s.trim().to_string();
                if let Some(tool_calls) = &choice.message.tool_calls {
                    if !tool_calls.is_empty() {
                        step_summary.push_str("\nTool calls: ");
                        for (i, tc) in tool_calls.iter().enumerate() {
                            if i > 0 {
                                step_summary.push_str(", ");
                            }
                            let args_preview = if tc.function.arguments.len() > 53 { // 50 for content + 3 for "..."
                                format!("{}...", &tc.function.arguments[..50])
                            } else {
                                tc.function.arguments.clone()
                            };
                            step_summary.push_str(&format!(
                                "{} (args: {})",
                                tc.function.name, args_preview
                            ));
                        }
                    }
                }
                step_summary
            })
        })
    })
}

async fn extract_task_in_step(
    llm_client: Arc<dyn LLMClient>,
    model_params: &LLMModelParameters,
    prev_step_str: &str,
    current_step_str: &str,
) -> Result<(String, String), LLMError> {
    let prompt = format!(
        "The following is an excerpt of the steps trying to solve a software bug by an AI agent: \
        <previous_step>{}</previous_step><this_step>{}</this_step>\n\n{}",
        prev_step_str, current_step_str, EXTRACTOR_PROMPT
    );

    let messages = vec![
        LLMMessage {
            role: MessageRole::User,
            content: Some(prompt),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
        LLMMessage {
            role: MessageRole::Assistant,
            content: Some(
                "Sure. Here is the task the agent is performing: <task>The agent".to_string(),
            ),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    let mut specific_params = model_params.clone();
    specific_params.temperature = 0.1;
    // TODO: If llm_client.chat needs to take ModelParameters per call for overrides like temp:
    // let response = llm_client.chat(messages.clone(), Some(specific_params), None, None).await?;
    // For now, assuming client uses its configured model and we just adjusted temp conceptually.
    // The actual LLMClient trait doesn't support per-call model param overrides beyond tools/tool_choice.

    for _attempt in 0..MAX_LAKEVIEW_RETRIES {
        let response = llm_client.chat(messages.clone(), None, None).await?;

        if let Some(content) = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
        {
            // Ensure a space between "The agent" and the LLM's actual content.
            // The LLM is prompted to continue after "The agent", so its content might start with " is..." or " calls..."
            let llm_provided_content = content.trim();
            let full_response_content = if llm_provided_content.starts_with("is") || llm_provided_content.starts_with("was") || llm_provided_content.starts_with("will") || !llm_provided_content.contains(" ") {
                // If it starts with a verb like "is" or is a single word, assume it's a direct continuation
                format!("<task>The agent {}", llm_provided_content)
            } else {
                // Otherwise, if it's more complex, it might already be a full phrase.
                // This part is a bit heuristic. The original format! might be fine if LLM always gives " is..."
                // Let's stick to a simpler fix first: ensure space.
                format!("<task>The agent {}", content.trim()) // Original fix attempt was this simple version
            };

            // Reverting to simpler, more direct fix: always add a space.
            // The prompt is "Sure. Here is the task the agent is performing: <task>The agent"
            // So the LLM content should be " is doing X..." or " calls Y..."
            // format!("<task>The agent{}", content.trim()) was the original failing line.
            // format!("<task>The agent {}", content.trim()) is the direct fix.
            let final_full_response_content = format!("<task>The agent {}", content.trim());

            if let (Some(task_start_idx), Some(task_end_idx)) = (
                final_full_response_content.find("<task>"),
                full_response_content.find("</task>"),
            ) {
                if let (Some(details_start_idx), Some(details_end_idx)) = (
                    full_response_content.find("<details>"),
                    full_response_content.find("</details>"),
                ) {
                    let task = full_response_content[task_start_idx + "<task>".len()..task_end_idx]
                        .trim()
                        .to_string();
                    let details = full_response_content
                        [details_start_idx + "<details>".len()..details_end_idx]
                        .trim()
                        .to_string();
                    if !task.is_empty() && !details.is_empty() {
                        return Ok((task, details));
                    }
                }
            }
            warn!(
                "Lakeview extract_task_in_step: Failed to parse task/details from LLM response: {}",
                full_response_content
            );
        } else {
            warn!("Lakeview extract_task_in_step: LLM response had no content.");
        }
    }
    Err(LLMError::ApiError(
        "Failed to extract task/details after multiple retries.".to_string(),
    ))
}

async fn extract_tags_in_step(
    llm_client: Arc<dyn LLMClient>,
    model_params: &LLMModelParameters,
    trajectory_so_far_str: &str,
    current_step_str: &str,
) -> Result<Vec<String>, LLMError> {
    let prompt = format!(
        "Below is the trajectory of an AI agent solving a software bug until the current step. Each step is marked within a <step> tag.\n\n{}\n\n<current_step>{}</current_step>\n\n{}",
        trajectory_so_far_str, current_step_str, TAGGER_PROMPT
    );

    let messages = vec![
        LLMMessage {
            role: MessageRole::User,
            content: Some(prompt),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
        LLMMessage {
            role: MessageRole::Assistant,
            content: Some("Sure. The tags are: <tags>".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        },
    ];

    let mut specific_params = model_params.clone();
    specific_params.temperature = 0.1;
    let known_tags_map = get_known_tags();

    for _attempt in 0..MAX_LAKEVIEW_RETRIES {
        // See TODO in extract_task_in_step about model_params per call
        let response = llm_client.chat(messages.clone(), None, None).await?;

        if let Some(content) = response
            .choices
            .first()
            .and_then(|c| c.message.content.as_ref())
        {
            let full_response_content = format!("<tags>{}", content.trim());

            if let (Some(tags_start_idx), Some(tags_end_idx)) = (
                full_response_content.find("<tags>"),
                full_response_content.find("</tags>"),
            ) {
                let tags_str =
                    full_response_content[tags_start_idx + "<tags>".len()..tags_end_idx].trim();
                let parsed_tags: Vec<String> = tags_str
                    .split(',')
                    .map(|s| s.trim().to_uppercase())
                    .filter(|s| !s.is_empty())
                    .collect();

                if !parsed_tags.is_empty()
                    && parsed_tags
                        .iter()
                        .all(|tag| known_tags_map.contains_key(tag.as_str()))
                {
                    return Ok(parsed_tags);
                } else if !parsed_tags.is_empty() {
                    warn!("Lakeview extract_tags_in_step: Parsed tags contain unknown tags: {:?}. Full response: {}", parsed_tags, full_response_content);
                } else {
                    warn!(
                        "Lakeview extract_tags_in_step: No valid tags parsed from: {}",
                        full_response_content
                    );
                }
            } else {
                warn!(
                    "Lakeview extract_tags_in_step: <tags> structure not found in LLM response: {}",
                    full_response_content
                );
            }
        } else {
            warn!("Lakeview extract_tags_in_step: LLM response had no content.");
        }
    }
    Err(LLMError::ApiError(
        "Failed to extract valid tags after multiple retries.".to_string(),
    ))
}

/// Generates a structured summary of an agent's execution trajectory.
///
/// This function iterates through each step of the provided `AgentExecution`,
/// calls helper functions (`extract_task_in_step`, `extract_tags_in_step`) to get
/// granular insights for each step, and then compiles these into a final summary string.
///
/// # Arguments
/// * `agent_execution`: A reference to the `AgentExecution` struct.
/// * `llm_client`: An `Arc<dyn LLMClient>` for making LLM calls for extraction/tagging.
/// * `summary_model_params`: `ModelParameters` for the LLM calls made by Lakeview functions.
///
/// # Returns
/// A `Result` containing the structured summary string or an `AgentError`.
pub async fn generate_summary(
    agent_execution: &AgentExecution,
    llm_client: Arc<dyn LLMClient>,
    summary_model_params: &LLMModelParameters,
) -> Result<String, AgentError> {
    info!(
        "Generating enhanced Lakeview summary for task: {}",
        agent_execution.task
    );

    if agent_execution.steps.is_empty() {
        warn!("Agent execution has no steps, cannot generate summary.");
        return Ok("No actions taken by the agent.".to_string());
    }

    let mut lakeview_steps_info: Vec<LakeViewStepInfo> = Vec::new();
    let mut trajectory_so_far_buffer = String::new();
    let known_tags_map = get_known_tags();

    for (i, step) in agent_execution.steps.iter().enumerate() {
        let prev_step_str = if i > 0 {
            format_agent_step_for_lakeview(&agent_execution.steps[i - 1]).unwrap_or_default()
        } else {
            "(none)".to_string()
        };
        let current_step_str = format_agent_step_for_lakeview(step)
            .unwrap_or_else(|| "No textual content for this step.".to_string());

        if current_step_str.trim().is_empty() && step.tool_calls_made.is_none() {
            // Skip empty steps for detailed analysis, but maybe log them generally
            lakeview_steps_info.push(LakeViewStepInfo {
                step_number: step.step_number,
                desc_task: "Internal or empty step".to_string(),
                desc_details: "".to_string(),
                tags_emoji: "".to_string(),
            });
            trajectory_so_far_buffer.push_str(&format!(
                "<step id=\"{}\">\n{}\n</step>\n\n",
                step.step_number, current_step_str
            ));
            continue;
        }

        let task_details_result = extract_task_in_step(
            llm_client.clone(),
            summary_model_params,
            &prev_step_str,
            &current_step_str,
        )
        .await;
        let (desc_task, desc_details) = match task_details_result {
            Ok((task, details)) => (task, details),
            Err(e) => {
                warn!(
                    "Lakeview: Failed to extract task/details for step {}: {:?}",
                    step.step_number, e
                );
                (
                    "Error extracting task".to_string(),
                    "Could not determine details.".to_string(),
                )
            }
        };

        let tags_result = extract_tags_in_step(
            llm_client.clone(),
            summary_model_params,
            &trajectory_so_far_buffer,
            &current_step_str,
        )
        .await;
        let tags_emoji = match tags_result {
            Ok(tags) => tags
                .iter()
                .map(|tag| known_tags_map.get(tag.as_str()).copied().unwrap_or("❓"))
                .collect::<Vec<&str>>()
                .join(" "),
            Err(e) => {
                warn!(
                    "Lakeview: Failed to extract tags for step {}: {:?}",
                    step.step_number, e
                );
                "❓".to_string()
            }
        };

        lakeview_steps_info.push(LakeViewStepInfo {
            step_number: step.step_number,
            desc_task,
            desc_details,
            tags_emoji,
        });
        trajectory_so_far_buffer.push_str(&format!(
            "<step id=\"{}\">\n{}\n</step>\n\n",
            step.step_number, current_step_str
        ));
    }

    // Format the final summary string from LakeViewStepInfo
    let mut final_summary = format!("Lakeview Summary for Task: \"{}\"\n", agent_execution.task);
    final_summary.push_str("------------------------------------\n");
    for lv_step in lakeview_steps_info {
        final_summary.push_str(&format!(
            "Step {}: {} {}\n  Details: {}\n",
            lv_step.step_number, lv_step.tags_emoji, lv_step.desc_task, lv_step.desc_details
        ));
    }
    final_summary.push_str("------------------------------------\n");
    final_summary.push_str(&format!(
        "Overall Task Success: {}\n",
        agent_execution.success
    ));
    if let Some(res) = &agent_execution.final_result {
        final_summary.push_str(&format!("Final Agent Message: {}\n", res));
    }
    if let Some(err) = &agent_execution.error_message {
        final_summary.push_str(&format!("Agent Error: {}\n", err));
    }

    Ok(final_summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::base_agent::{AgentState, AgentStep};
    use crate::config::ModelParameters as ConfigModelParameters;
    use crate::llm::base_client::{
        LLMMessage, LLMResponse, LLMResponseChoice, MessageRole, ToolCall as LLMToolCall,
        ToolCallFunction,
    };
    use crate::llm::OpenAIClient;

    use serde_json::json;
    use wiremock::matchers::{body_partial_json, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate}; // Removed Times

    fn create_dummy_execution() -> AgentExecution {
        AgentExecution {
            task: "Test summarization".to_string(),
            start_time: 0,
            end_time: Some(10),
            steps: vec![
                AgentStep {
                    step_number: 1,
                    state: AgentState::Thinking,
                    messages_to_llm: Some(vec![LLMMessage {
                        role: MessageRole::User,
                        content: Some("User input for step 1".to_string()),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    }]),
                    llm_response: Some(LLMResponse {
                        id: "resp1".to_string(),
                        object: "chat.completion".to_string(),
                        created: 1,
                        model: "gpt-test".to_string(),
                        choices: vec![LLMResponseChoice {
                            index: 0,
                            message: LLMMessage {
                                role: MessageRole::Assistant,
                                content: Some(
                                    "LLM thought for step 1 and decided to use a tool.".to_string(),
                                ),
                                name: None,
                                tool_calls: Some(vec![LLMToolCall {
                                    id: "tool_call_123".to_string(),
                                    tool_type: "function".to_string(),
                                    function: ToolCallFunction {
                                        name: "example_tool".to_string(),
                                        arguments: "{\"foo\":\"bar\"}".to_string(),
                                    },
                                }]),
                                tool_call_id: None,
                            },
                            finish_reason: Some("tool_calls".to_string()),
                        }],
                        usage: None,
                    }),
                    tool_calls_made: None,
                    tool_results: None,
                    reflection: None,
                    error: None,
                    duration_ms: 100,
                },
                AgentStep {
                    // Add a second step for more comprehensive summary testing
                    step_number: 2,
                    state: AgentState::ProcessingToolResult,
                    messages_to_llm: Some(vec![/* ... */]), // Can be simplified for test if not directly used by format_agent_step
                    llm_response: Some(LLMResponse {
                        id: "resp2".to_string(),
                        object: "chat.completion".to_string(),
                        created: 1,
                        model: "gpt-test".to_string(),
                        choices: vec![LLMResponseChoice {
                            index: 0,
                            message: LLMMessage {
                                role: MessageRole::Assistant,
                                content: Some("LLM thought after tool for step 2".to_string()),
                                name: None,
                                tool_calls: None,
                                tool_call_id: None,
                            },
                            finish_reason: Some("stop".to_string()),
                        }],
                        usage: None,
                    }),
                    tool_calls_made: None,
                    tool_results: Some(vec![crate::tools::AgentToolResult {
                        tool_call_id: "tool_call_123".to_string(),
                        success: true,
                        result: Some("Tool output".to_string()),
                        error: None,
                    }]),
                    reflection: None,
                    error: None,
                    duration_ms: 50,
                },
            ],
            final_result: Some("Task done.".to_string()),
            success: true,
            total_tokens_used: Some(crate::llm::base_client::LLMUsage { // Corrected type
                prompt_tokens: 50,
                completion_tokens: Some(0),
                total_tokens: 50,
            }),
            error_message: None,
        }
    }

    fn get_lakeview_model_params() -> ConfigModelParameters {
        ConfigModelParameters {
            api_key: Some("test_lakeview_key".to_string()),
            model: "gpt-3.5-turbo-instruct".to_string(),
            max_tokens: Some(MAX_SUMMARY_TOKENS), // Use const for consistency
            temperature: 0.1,                     // Default for lakeview calls
            top_p: 1.0,
            top_k: None,
            parallel_tool_calls: false,
            max_retries: crate::config::default_max_retries(),
            base_url: None,
            api_version: None,
            candidate_count: None,
            stop_sequences: None,
        }
    }

    #[tokio::test]
    async fn test_generate_summary_with_extraction_and_tagging() {
        let server = MockServer::start().await;

        // Mock for first step's extract_task_in_step
        let step1_tool_args_str = "{\"foo\":\"bar\"}";
        let step1_current_step_str_formatted_for_mock = format!(
            "LLM thought for step 1 and decided to use a tool.\nTool calls: example_tool (args: {})",
            step1_tool_args_str
        );
        let step1_task_extraction_prompt_for_mock = format!(
            "The following is an excerpt of the steps trying to solve a software bug by an AI agent: <previous_step>(none)</previous_step><this_step>{}</this_step>\n\n{}",
            step1_current_step_str_formatted_for_mock, EXTRACTOR_PROMPT
        );

        Mock::given(method("POST")).and(path("/chat/completions"))
            .and(body_partial_json(json!({
                "messages": [{
                    "role": "user",
                    "content": step1_task_extraction_prompt_for_mock // Use the precisely formatted string
                }]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "extract_resp1", "object": "chat.completion", "created": 124, "model": "gpt-test",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": " is calling a tool.</task><details>Agent calls example_tool for step 1.</details>"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            })))
            .expect(1) // Expect once for step 1 task extraction
            .mount(&server)
            .await;

        // Mock for first step's extract_tags_in_step
        // The trajectory_so_far_str for the first step's tag extraction will be empty.
        // current_step_str will be step1_current_step_str_formatted_for_mock.
        let step1_tag_extraction_prompt_for_mock = format!(
            "Below is the trajectory of an AI agent solving a software bug until the current step. Each step is marked within a <step> tag.\n\n\n\n<current_step>{}</current_step>\n\n{}",
            step1_current_step_str_formatted_for_mock, TAGGER_PROMPT
        );
        Mock::given(method("POST")).and(path("/chat/completions"))
             .and(body_partial_json(json!({
                 "messages": [{
                     "role": "user",
                     "content": step1_tag_extraction_prompt_for_mock
                 }]
             })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "tag_resp1", "object": "chat.completion", "created": 125, "model": "gpt-test",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "THINK,OUTLIER</tags>"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            })))
            .expect(1) // Expect once for step 1 tag extraction
            .mount(&server)
            .await;

        // Mock for second step's extract_task_in_step
        let step2_prev_step_str_for_mock = step1_current_step_str_formatted_for_mock.clone();
        let step2_current_step_str_for_mock = "LLM thought after tool for step 2".to_string();
        let step2_task_extraction_prompt_for_mock = format!(
            "The following is an excerpt of the steps trying to solve a software bug by an AI agent: <previous_step>{}</previous_step><this_step>{}</this_step>\n\n{}",
            step2_prev_step_str_for_mock, step2_current_step_str_for_mock, EXTRACTOR_PROMPT
        );

        Mock::given(method("POST")).and(path("/chat/completions"))
            .and(body_partial_json(json!({
                "messages": [{
                    "role": "user",
                    "content": step2_task_extraction_prompt_for_mock
                }]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "extract_resp2", "object": "chat.completion", "created": 126, "model": "gpt-test",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": " is processing tool results.</task><details>Agent processes results for step 2.</details>"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            })))
            .expect(1)
            .mount(&server)
            .await;

        // Mock for second step's extract_tags_in_step
        let step2_trajectory_so_far_for_mock = format!(
            "<step id=\"1\">\n{}\n</step>\n\n", // Step ID is 1 from dummy execution
            step1_current_step_str_formatted_for_mock
        );
        let step2_tag_extraction_prompt_for_mock = format!(
            "Below is the trajectory of an AI agent solving a software bug until the current step. Each step is marked within a <step> tag.\n\n{}\n\n<current_step>{}</current_step>\n\n{}",
            step2_trajectory_so_far_for_mock, step2_current_step_str_for_mock, TAGGER_PROMPT
        );
        Mock::given(method("POST")).and(path("/chat/completions"))
            .and(body_partial_json(json!({
                "messages": [{
                    "role": "user",
                    "content": step2_tag_extraction_prompt_for_mock
                }]
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "tag_resp2", "object": "chat.completion", "created": 127, "model": "gpt-test",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "THINK</tags>"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let exec = create_dummy_execution();
        let model_params = get_lakeview_model_params();
        let llm_client = Arc::new(
            OpenAIClient::new(
                model_params.api_key.clone(),
                Some(server.uri()),
                model_params.clone(),
            )
            .await
            .unwrap(),
        );

        let summary_result = generate_summary(&exec, llm_client, &model_params).await;
        assert!(
            summary_result.is_ok(),
            "generate_summary failed: {:?}",
            summary_result.err()
        );
        let summary = summary_result.unwrap();

        println!("Generated Summary:\n{}", summary); // For manual inspection

        assert!(summary.contains("Step 1: 🧠 ⁉️ The agent is calling a tool."));
        assert!(summary.contains("Details: Agent calls example_tool for step 1."));
        assert!(summary.contains("Step 2: 🧠 The agent is processing tool results."));
        assert!(summary.contains("Details: Agent processes results for step 2."));
        assert!(summary.contains("Overall Task Success: true"));
        assert!(summary.contains("Final Agent Message: Task done."));
    }

    #[tokio::test]
    async fn test_generate_summary_empty_steps() {
        let mut exec = create_dummy_execution();
        exec.steps = vec![];
        let model_params = get_lakeview_model_params();
        let server = MockServer::start().await;
        let llm_client_for_empty_test = Arc::new(
            OpenAIClient::new(
                Some("dummykey".to_string()),
                Some(server.uri()),
                model_params.clone(),
            )
            .await
            .unwrap(),
        );
        let result = generate_summary(&exec, llm_client_for_empty_test, &model_params).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "No actions taken by the agent.");
    }

    #[tokio::test]
    async fn test_extract_task_in_step_parsing() {
        let server = MockServer::start().await;
        Mock::given(method("POST")).and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "extract_resp", "object": "chat.completion", "created": 124, "model": "gpt-test",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": " is examining code.</task><details>The agent is looking at file.py for 'foo_bar' function.</details>"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            })))
            .mount(&server).await;

        let model_params = get_lakeview_model_params();
        let llm_client = Arc::new(
            OpenAIClient::new(
                model_params.api_key.clone(),
                Some(server.uri()),
                model_params.clone(),
            )
            .await
            .unwrap(),
        );

        let result = extract_task_in_step(
            llm_client,
            &model_params,
            "Previous step info",
            "Current step info",
        )
        .await;
        assert!(result.is_ok());
        let (task, details) = result.unwrap();
        assert_eq!(task, "The agent is examining code.");
        assert_eq!(
            details,
            "The agent is looking at file.py for 'foo_bar' function."
        );
    }

    #[tokio::test]
    async fn test_extract_tags_in_step_parsing() {
        let server = MockServer::start().await;
        Mock::given(method("POST")).and(path("/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "extract_tags_resp", "object": "chat.completion", "created": 124, "model": "gpt-test",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "EXAMINE_CODE,THINK</tags>"}, "finish_reason": "stop"}], // Note: No leading space before EXAMINE_CODE
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
            })))
            .mount(&server).await;

        let model_params = get_lakeview_model_params();
        let llm_client = Arc::new(
            OpenAIClient::new(
                model_params.api_key.clone(),
                Some(server.uri()),
                model_params.clone(),
            )
            .await
            .unwrap(),
        );

        let result = extract_tags_in_step(
            llm_client,
            &model_params,
            "Trajectory so far...",
            "Current step info",
        )
        .await;
        assert!(
            result.is_ok(),
            "extract_tags_in_step failed: {:?}",
            result.err()
        );
        let tags = result.unwrap();
        assert_eq!(tags, vec!["EXAMINE_CODE".to_string(), "THINK".to_string()]);
    }
}
