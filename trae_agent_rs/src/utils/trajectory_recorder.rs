// trae-agent-rust/src/utils/trajectory_recorder.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::{Context, Result}; // Using anyhow for error handling

use crate::agent::base_agent::AgentStep; // Removed AgentState
use crate::llm::base_client::LLMUsage; // Removed LLMMessage, LLMResponse

// Mirroring Python's TrajectoryHeader
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TrajectoryHeader {
    pub version: String,
    pub task: String,
    pub provider: String,
    pub model: String,
    pub max_steps: u32,
    pub timestamp: u64, // Unix timestamp
    pub extra_args: Option<HashMap<String, String>>,
}

// Mirroring Python's Trajectory (simplified, as steps are recorded incrementally)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Trajectory {
    pub header: TrajectoryHeader,
    pub steps: Vec<AgentStep>, // AgentStep will be adapted to match Python's recorded structure
    pub success: bool,
    pub final_result: Option<String>,
    pub total_tokens: Option<LLMUsage>, // Assuming LLMUsage holds token counts
}

pub struct TrajectoryRecorder {
    trajectory_path: PathBuf,
    trajectory: Option<Trajectory>, // Holds the current trajectory being built
    writer: Option<BufWriter<File>>, // For writing incrementally if needed, or just at the end
}

impl TrajectoryRecorder {
    const TRAJECTORY_VERSION: &'static str = "1.0";

    /// Creates a new TrajectoryRecorder.
    /// If `trajectory_path` is None, a default path is generated.
    pub fn new(trajectory_path: Option<PathBuf>) -> Result<Self> {
        let path = trajectory_path.unwrap_or_else(|| {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            PathBuf::from(format!("trajectory_{}.json", now))
        });

        // Ensure parent directory exists
        if let Some(parent_dir) = path.parent() {
            if !parent_dir.exists() {
                std::fs::create_dir_all(parent_dir)
                    .with_context(|| format!("Failed to create directory: {:?}", parent_dir))?;
            }
        }

        Ok(Self {
            trajectory_path: path,
            trajectory: None,
            writer: None, // Initialize writer later if needed for incremental writes
        })
    }

    /// Gets the path where the trajectory will be saved.
    pub fn get_trajectory_path(&self) -> &Path {
        &self.trajectory_path
    }

    /// Starts recording a new trajectory.
    pub fn start_recording(
        &mut self,
        task: String,
        provider: String,
        model: String,
        max_steps: u32,
        extra_args: Option<HashMap<String, String>>,
    ) -> Result<()> {
        if self.trajectory.is_some() {
            // Finalize previous recording if any, though ideally this shouldn't happen without finalize_recording being called
            self.finalize_recording(false, None, None)?;
        }

        let header = TrajectoryHeader {
            version: Self::TRAJECTORY_VERSION.to_string(),
            task,
            provider,
            model,
            max_steps,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            extra_args,
        };

        self.trajectory = Some(Trajectory {
            header,
            steps: Vec::new(),
            success: false,
            final_result: None,
            total_tokens: None,
        });

        // If we want to write incrementally, setup writer here.
        // For now, we'll write everything in finalize_recording.
        tracing::info!("Trajectory recording started for path: {:?}", self.trajectory_path);
        Ok(())
    }

    /// Records a single agent step.
    /// This should be adapted to take parameters similar to Python's `record_agent_step`.
    /// For now, it takes a pre-constructed AgentStep.
    pub fn record_agent_step(&mut self, step: AgentStep) {
        if let Some(trajectory) = self.trajectory.as_mut() {
            trajectory.steps.push(step);
        } else {
            tracing::warn!("Attempted to record step, but trajectory recording was not started.");
        }
    }
    // Removed unused record_llm_interaction method

    /// Finalizes the recording, writing the trajectory to the file.
    pub fn finalize_recording(
        &mut self,
        success: bool,
        final_result: Option<String>,
        total_tokens: Option<LLMUsage>, // Or whatever type represents token usage
    ) -> Result<()> {
        if let Some(mut trajectory) = self.trajectory.take() {
            trajectory.success = success;
            trajectory.final_result = final_result;
            trajectory.total_tokens = total_tokens;

            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true) // Overwrite if exists
                .open(&self.trajectory_path)
                .with_context(|| format!("Failed to open trajectory file for writing: {:?}", self.trajectory_path))?;

            let mut writer = BufWriter::new(file);
            serde_json::to_writer_pretty(&mut writer, &trajectory)
                .with_context(|| format!("Failed to serialize trajectory to JSON: {:?}", self.trajectory_path))?;
            writer.flush().with_context(|| format!("Failed to flush trajectory writer: {:?}", self.trajectory_path))?;

            tracing::info!("Trajectory finalized and saved to: {:?}", self.trajectory_path);
        } else {
            tracing::warn!("Attempted to finalize recording, but no trajectory was active.");
        }
        self.writer = None; // Ensure writer is cleared
        Ok(())
    }
}

// Example AgentStep structure that might be in base_agent.rs
// Ensure this matches the actual AgentStep definition used.
/*
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AgentStep {
    pub step_number: u32,
    pub state: AgentState, // Assuming AgentState is serializable
    pub messages_to_llm: Option<Vec<LLMMessage>>, // Assuming LLMMessage is serializable
    pub llm_response: Option<LLMResponse>, // Assuming LLMResponse is serializable
    pub tool_calls_made: Option<Vec<crate::llm::base_client::ToolCall>>, // Assuming ToolCall is serializable
    pub tool_results: Option<Vec<crate::tools::AgentToolResult>>, // Assuming AgentToolResult is serializable
    pub reflection: Option<String>,
    pub error: Option<String>,
    pub duration_ms: u128,
}
*/

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use crate::agent::base_agent::{AgentState, AgentStep}; // Ensure correct path
    use crate::llm::base_client::{LLMMessage, LLMResponse, MessageRole, LLMResponseChoice, LLMUsage}; // Corrected imports

    // Mock AgentStep for testing - ensure this aligns with your actual AgentStep
    // If your AgentStep is already in scope and serializable, you might not need this mock.
    /*
    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct MockAgentStep {
        step_number: u32,
        state: String, // Simplified state for mock
        action: String,
    }
    */


    fn create_dummy_agent_step(step_num: u32) -> AgentStep {
        AgentStep {
            step_number: step_num,
            state: AgentState::Thinking, // Example state
            messages_to_llm: Some(vec![LLMMessage {
                role: MessageRole::User,
                content: Some("Hello".to_string()),
                name: None,
                tool_calls: None,
                tool_call_id: None,
            }]),
            llm_response: Some(LLMResponse {
                id: "resp_id".to_string(),
                object: "chat.completion".to_string(),
                created: 0,
                model: "test-model".to_string(),
                choices: vec![LLMResponseChoice { // Corrected type
                    index: 0,
                    message: LLMMessage { // Corrected type, ResponseMessage is not a direct struct here
                        role: MessageRole::Assistant,
                        content: Some("Hi there".to_string()),
                        name: None,
                        tool_calls: None,
                        tool_call_id: None,
                    },
                    finish_reason: Some("stop".to_string()),
                }],
                usage: Some(LLMUsage {
                    prompt_tokens: 10,
                    completion_tokens: Some(5),
                    total_tokens: 15,
                }),
                // system_fingerprint: None, // Removed, does not exist on LLMResponse
            }),
            tool_calls_made: None,
            tool_results: None,
            reflection: None,
            error: None,
            duration_ms: 100,
        }
    }


    #[test]
    fn test_trajectory_recorder_new_default_path() {
        let recorder = TrajectoryRecorder::new(None).unwrap();
        assert!(recorder.get_trajectory_path().to_string_lossy().contains("trajectory_"));
        assert!(recorder.get_trajectory_path().to_string_lossy().ends_with(".json"));
    }

    #[test]
    fn test_trajectory_recorder_new_custom_path() {
        let dir = tempdir().unwrap();
        let custom_path = dir.path().join("my_custom_trajectory.json");
        let recorder = TrajectoryRecorder::new(Some(custom_path.clone())).unwrap();
        assert_eq!(recorder.get_trajectory_path(), custom_path);
        // Ensure the directory was created if it didn't exist (tempdir creates it)
        assert!(custom_path.parent().unwrap().exists());
    }

    #[test]
    fn test_trajectory_recording_flow() -> Result<()> {
        let dir = tempdir().unwrap();
        let trajectory_file = dir.path().join("test_flow_trajectory.json");

        let mut recorder = TrajectoryRecorder::new(Some(trajectory_file.clone()))?;

        recorder.start_recording(
            "Test Task".to_string(),
            "test_provider".to_string(),
            "test_model".to_string(),
            10,
            None,
        )?;

        recorder.record_agent_step(create_dummy_agent_step(1));
        recorder.record_agent_step(create_dummy_agent_step(2));

        let final_tokens = LLMUsage { prompt_tokens: 100, completion_tokens: Some(50), total_tokens: 150 };
        recorder.finalize_recording(true, Some("Task completed successfully".to_string()), Some(final_tokens.clone()))?;

        assert!(trajectory_file.exists());

        let file_content = std::fs::read_to_string(&trajectory_file)?;
        let saved_trajectory: Trajectory = serde_json::from_str(&file_content)?;

        assert_eq!(saved_trajectory.header.task, "Test Task");
        assert_eq!(saved_trajectory.header.provider, "test_provider");
        assert_eq!(saved_trajectory.steps.len(), 2);
        assert_eq!(saved_trajectory.steps[0].step_number, 1);
        assert!(saved_trajectory.success);
        assert_eq!(saved_trajectory.final_result, Some("Task completed successfully".to_string()));
        assert_eq!(saved_trajectory.total_tokens.unwrap().total_tokens, final_tokens.total_tokens);

        Ok(())
    }

    #[test]
    fn test_trajectory_recorder_create_parent_dirs() -> Result<()> {
        let dir = tempdir().unwrap();
        let nested_path = dir.path().join("nested").join("dir").join("deep_trajectory.json");

        // nested_path.parent() itself doesn't exist yet
        assert!(!nested_path.parent().unwrap().exists());

        let _recorder = TrajectoryRecorder::new(Some(nested_path.clone()))?;

        // Check that the parent directory was created by TrajectoryRecorder::new
        assert!(nested_path.parent().unwrap().exists());

        Ok(())
    }
}
