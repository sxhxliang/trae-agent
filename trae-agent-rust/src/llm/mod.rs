//! # LLM Module
//!
//! Provides abstractions and clients for interacting with Large Language Models (LLMs).
//! It defines a common `LLMClient` trait and implementations for specific providers
//! like OpenAI.

pub mod anthropic_client;
pub mod base_client;
pub mod openai_client;

pub use anthropic_client::AnthropicClient;
pub use base_client::{
    LLMClient, LLMError, LLMMessage, MessageRole, ModelParameters as LLMModelParameters,
};
pub use openai_client::OpenAIClient;
