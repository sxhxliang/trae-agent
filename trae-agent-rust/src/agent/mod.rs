//! # Agent Module
//!
//! Contains the core logic for the AI agent, including the `Agent` trait defining
//! agent capabilities, `BaseAgent` for common state and execution loop infrastructure,
//! and `TraeAgent` as the specific implementation for software engineering tasks.

pub mod base_agent;
pub mod trae_agent_rs; // trae_agent_rs to avoid conflict with potential crate name

pub use base_agent::{Agent, AgentError, AgentExecution};
pub use trae_agent_rs::TraeAgent;
