use super::base_client::{
    LLMClient,
    LLMError,
    LLMMessage,
    LLMResponse,
    ModelParameters,
    ToolChoice,
    ToolDefinition, // Removed ToolCall
};
use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Client as HttpClient;
use serde::Serialize;
use tracing::{debug, error, instrument};

const DEFAULT_OPENAI_API_BASE: &str = "https://api.openai.com/v1";

#[derive(Serialize, Debug)]
struct OpenAIChatRequest<'a> {
    model: &'a str,
    messages: &'a [LLMMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [ToolDefinition]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    // Add other parameters like stream, n, stop, presence_penalty, frequency_penalty, logit_bias, user if needed
}

#[derive(Debug)]
pub struct OpenAIClient {
    http_client: HttpClient,
    #[allow(dead_code)] // Set in new, but primarily used for Authorization header setup.
    api_key: String,
    base_url: String,
    model_parameters: ModelParameters,
}

#[async_trait] // Added
impl LLMClient for OpenAIClient {
    #[instrument(skip(api_key, model_parameters))]
    async fn new(
        api_key: Option<String>,
        base_url: Option<String>,
        model_parameters: ModelParameters,
    ) -> Result<Self, LLMError> {
        let key = api_key.or_else(|| std::env::var("OPENAI_API_KEY").ok());
        if key.is_none() {
            error!("OpenAI API key not provided and not found in OPENAI_API_KEY env var.");
            return Err(LLMError::NoApiKey);
        }
        let final_key = key.unwrap();

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", final_key))
                .map_err(|e| LLMError::Other(format!("Invalid API key format: {}", e)))?,
        );
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let http_client = HttpClient::builder()
            .default_headers(headers)
            .build()
            .map_err(LLMError::Network)?;

        Ok(Self {
            http_client,
            api_key: final_key, // Stored for potential future use, though already in headers
            base_url: base_url.unwrap_or_else(|| DEFAULT_OPENAI_API_BASE.to_string()),
            model_parameters,
        })
    }

    #[instrument(skip(self, messages, tools))]
    async fn chat(
        &self,
        messages: Vec<LLMMessage>,
        tools: Option<Vec<ToolDefinition>>,
        tool_choice: Option<ToolChoice>,
    ) -> Result<LLMResponse, LLMError> {
        let request_payload = OpenAIChatRequest {
            model: &self.model_parameters.model,
            messages: &messages,
            tools: tools.as_deref(),
            tool_choice: tool_choice.as_ref(),
            temperature: Some(self.model_parameters.temperature),
            top_p: Some(self.model_parameters.top_p),
            max_tokens: self.model_parameters.max_tokens,
        };

        debug!(payload = ?request_payload, "Sending OpenAI chat request");

        let url = format!("{}/chat/completions", self.base_url);
        let response = self
            .http_client
            .post(&url)
            .json(&request_payload)
            .send()
            .await
            .map_err(LLMError::Network)?;

        let status = response.status(); // Store status first
        debug!(status = ?status, "Received OpenAI response status");

        if !status.is_success() {
            let error_body = response.text().await.map_err(LLMError::Network)?;
            error!(error_body = %error_body, "OpenAI API error");
            return Err(LLMError::ApiError(format!(
                "API request failed with status {}: {}",
                status, // Use stored status
                error_body
            )));
        }

        let llm_response = response.json::<LLMResponse>().await.map_err(|e| {
            error!(error = %e, "Failed to parse OpenAI JSON response");
            // LLMError::ParsingError expects serde_json::Error. reqwest::Error can be other things.
            // If e.is_decode() is true, it's a JSON parsing issue. Otherwise, network.
            if e.is_decode() {
                // Attempt to get underlying serde error if possible, or just use reqwest error string
                LLMError::Other(format!("JSON decoding error: {}", e))
            } else {
                LLMError::Network(e)
            }
        })?;

        debug!(response_id = %llm_response.id, "Successfully parsed OpenAI response");
        Ok(llm_response)
    }

    fn get_provider_name(&self) -> String {
        "openai".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::base_client::{LLMMessage, MessageRole}; // Removed ToolCallFunction
    use serde_json::json;
    use wiremock::matchers::{bearer_token, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    async fn setup_mock_server(api_key: &str) -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/chat/completions")) // Removed /v1
            .and(bearer_token(api_key))
            .and(header("Content-Type", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-test123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-4-test",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello there!"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21
                }
            })))
            .mount(&server)
            .await;
        server
    }

    fn get_default_model_params() -> ModelParameters {
        ModelParameters {
            api_key: None, // API key is passed directly to client new()
            model: "gpt-4-test".to_string(),
            max_tokens: Some(100),
            temperature: 0.7,
            top_p: 1.0,
            top_k: None,
            parallel_tool_calls: true,
            max_retries: crate::config::default_max_retries(), // Assuming this is accessible
            base_url: None,
            api_version: None,
            candidate_count: None,
            stop_sequences: None,
        }
    }

    #[tokio::test]
    async fn test_openai_client_new_success() {
        let api_key = "test_api_key".to_string();
        let params = get_default_model_params();
        let client_result = OpenAIClient::new(Some(api_key.clone()), None, params).await;
        assert!(client_result.is_ok());
    }

    #[tokio::test]
    async fn test_openai_client_new_no_api_key() {
        std::env::remove_var("OPENAI_API_KEY"); // Ensure env var is not set
        let params = get_default_model_params();
        let client_result = OpenAIClient::new(None, None, params).await;
        assert!(client_result.is_err());
        match client_result.err().unwrap() {
            LLMError::NoApiKey => {} // Expected
            _ => panic!("Expected NoApiKey error"),
        }
    }

    #[tokio::test]
    async fn test_openai_chat_simple_message() {
        let api_key = "test_api_key_chat";
        let server = setup_mock_server(api_key).await;
        let params = get_default_model_params();

        let client = OpenAIClient::new(Some(api_key.to_string()), Some(server.uri()), params)
            .await
            .unwrap();

        let messages = vec![LLMMessage {
            role: MessageRole::User,
            content: Some("Hello".to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let response = client.chat(messages, None, None).await.unwrap();
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("Hello there!")
        );
        assert_eq!(response.model, "gpt-4-test");
    }

    #[tokio::test]
    async fn test_openai_chat_with_tools() {
        let api_key = "test_api_key_tools";
        let server = MockServer::start().await;

        let tool_def = ToolDefinition {
            tool_type: "function".to_string(),
            function: super::super::base_client::FunctionDefinition {
                name: "get_weather".to_string(),
                description: "Get current weather".to_string(),
                parameters: super::super::base_client::FunctionParameters {
                    param_type: "object".to_string(),
                    properties: std::collections::HashMap::new(),
                    required: vec![],
                },
            },
        };

        Mock::given(method("POST"))
            .and(path("/chat/completions")) // Removed /v1
            .and(bearer_token(api_key))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "id": "chatcmpl-toolstest",
                "object": "chat.completion",
                "created": 1677652289,
                "model": "gpt-4-tool-test",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\": \"Boston\"}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": { "prompt_tokens": 10, "total_tokens": 20 }
            })))
            .mount(&server)
            .await;

        let params = get_default_model_params();
        let client = OpenAIClient::new(Some(api_key.to_string()), Some(server.uri()), params)
            .await
            .unwrap();

        let messages = vec![LLMMessage {
            role: MessageRole::User,
            content: Some("What's the weather in Boston?".to_string()), // Wrapped in Some()
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }];

        let response = client
            .chat(messages, Some(vec![tool_def]), None)
            .await
            .unwrap();
        assert!(response.choices[0].message.content.is_none());
        assert!(response.choices[0].message.tool_calls.is_some());
        let tool_calls = response.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].function.name, "get_weather");
        assert_eq!(
            tool_calls[0].function.arguments,
            "{\"location\": \"Boston\"}"
        );
    }
}
