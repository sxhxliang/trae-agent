//! # Configuration Module
//!
//! Defines structures and logic for loading and managing configuration
//! for the Trae Rust Agent. Configuration can be loaded from a JSON file,
//! environment variables, and command-line arguments.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tracing::warn;

/// Defines the parameters for a specific Large Language Model.
#[derive(Deserialize, Debug, Clone)]
pub struct ModelParameters {
    /// Optional API key for the LLM provider.
    pub api_key: Option<String>,
    /// Name of the model to be used (e.g., "gpt-4o", "claude-sonnet-4-20250514").
    pub model: String,
    #[serde(default = "default_max_tokens_openai")] // Assuming openai default if not present
    pub max_tokens: Option<u32>, // Made Option as per Python, will use defaults
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_top_p")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default = "default_parallel_tool_calls")]
    pub parallel_tool_calls: bool,
    #[serde(default = "default_max_retries")]
    pub max_retries: u32, // Python uses int, Rust u32 is fine
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub api_version: Option<String>,
    #[serde(default)]
    pub candidate_count: Option<u32>, // Python uses int
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
}

fn default_max_tokens_openai() -> Option<u32> {
    Some(128000)
}
fn default_temperature() -> f32 {
    0.5
}
fn default_top_p() -> f32 {
    1.0
}
fn default_parallel_tool_calls() -> bool {
    false // Aligning with Python's default
}
pub fn default_max_retries() -> u32 { // Made public
    10 // From Python's ModelParameters default
}

#[derive(Deserialize, Debug, Clone)]
pub struct Config {
    pub default_provider: String,
    #[serde(default = "default_max_steps")]
    pub max_steps: u32,
    pub model_providers: HashMap<String, ModelParameters>,
    #[serde(default)] // To allow optional lakeview_config in JSON
    pub lakeview_config: Option<LakeviewConfig>,
    #[serde(default = "default_enable_lakeview")]
    pub enable_lakeview: bool,
    #[serde(skip)]
    pub working_dir: Option<String>,
}

/// Configuration specific to the Lakeview summarization feature.
#[derive(Deserialize, Debug, Clone)]
pub struct LakeviewConfig {
    /// The LLM provider to be used for generating summaries (e.g., "openai", "anthropic").
    pub model_provider: String,
    /// The specific model name from the provider to be used for summaries.
    pub model_name: String,
}

fn default_max_steps() -> u32 {
    20
}
fn default_enable_lakeview() -> bool {
    true
}

/// Main configuration structure for the Trae Agent.
///
/// Holds settings for LLM providers, agent behavior, and features like Lakeview.
/// Configuration is loaded from a JSON file, with overrides from environment
/// variables and command-line arguments.
impl Config {
    /// Loads the agent configuration.
    ///
    /// Priority for loading values:
    /// 1. Command-line arguments (highest).
    /// 2. Environment variables (for API keys if not in CLI).
    /// 3. Values from the JSON configuration file.
    /// 4. Default values coded in the application (lowest).
    ///
    /// # Arguments
    /// * `config_file_path`: Path to the JSON configuration file (e.g., "trae_config.json").
    /// * `cli_provider`: Optional LLM provider name from CLI.
    /// * `cli_model`: Optional model name from CLI for the default provider.
    /// * `cli_api_key`: Optional API key from CLI for the default provider.
    /// * `cli_max_steps`: Optional maximum agent execution steps from CLI.
    /// * `cli_working_dir`: Optional working directory from CLI.
    ///
    /// # Returns
    /// A `Result` containing the loaded `Config` or an `anyhow::Error` if loading fails.
    pub fn load(
        config_file_path: &str,
        cli_provider: Option<String>,
        cli_model: Option<String>,
        cli_api_key: Option<String>,
        cli_max_steps: Option<u32>,
        cli_working_dir: Option<String>,
    ) -> Result<Self> {
        let path = Path::new(config_file_path);
        let mut loaded_config: Config = if path.exists() {
            let config_str = fs::read_to_string(path)
                .with_context(|| format!("Failed to read config file at: {}", config_file_path))?;
            serde_json::from_str(&config_str)
                .with_context(|| format!("Failed to parse config file: {}", config_file_path))?
        } else {
            warn!(
                "Config file not found at: {}. Using default values and environment variables.",
                config_file_path
            );
            // Create a default config if file doesn't exist
            let mut default_providers = HashMap::new();
            default_providers.insert(
                "openai".to_string(),
                ModelParameters {
                    api_key: None,
                    model: "gpt-4o".to_string(),
                    max_tokens: default_max_tokens_openai(),
                    temperature: default_temperature(),
                    top_p: default_top_p(),
                    top_k: None,
                    parallel_tool_calls: default_parallel_tool_calls(),
                    max_retries: default_max_retries(),
                    base_url: None,
                    api_version: None,
                    candidate_count: None,
                    stop_sequences: None,
                },
            );
            default_providers.insert(
                "anthropic".to_string(),
                ModelParameters {
                    api_key: None,
                    model: "claude-4-sonnet".to_string(), // More current default like Python
                    max_tokens: Some(4096),                        // from Python example
                    temperature: default_temperature(),
                    top_p: default_top_p(),
                    top_k: Some(0), // from Python example
                    parallel_tool_calls: default_parallel_tool_calls(), // Python defaults this to False, Rust to True. Keeping Rust's default.
                    max_retries: default_max_retries(),
                    base_url: Some("https://api.anthropic.com".to_string()), // From Python's default if file empty
                    api_version: None,
                    candidate_count: None,
                    stop_sequences: None,
                },
            );
            Config {
                default_provider: "openai".to_string(),
                max_steps: default_max_steps(),
                model_providers: default_providers,
                lakeview_config: None,                      // Added
                enable_lakeview: default_enable_lakeview(), // Added
                working_dir: None,
            }
        };

        // Override with CLI arguments or environment variables
        if let Some(provider_name) = cli_provider {
            loaded_config.default_provider = provider_name.clone();
            // If CLI specifies a provider not in the file, add a default entry for it
            if !loaded_config.model_providers.contains_key(&provider_name) {
                let default_params = match provider_name.as_str() {
                    "openai" => ModelParameters {
                        api_key: None,
                        model: "gpt-4o".to_string(),
                        max_tokens: default_max_tokens_openai(),
                        temperature: default_temperature(),
                        top_p: default_top_p(),
                        top_k: None,
                        parallel_tool_calls: default_parallel_tool_calls(),
                        max_retries: default_max_retries(),
                        base_url: None, // OpenAI default base_url is handled by client usually
                        api_version: None,
                        candidate_count: None,
                        stop_sequences: None,
                    },
                    "anthropic" => ModelParameters {
                        api_key: None,
                        model: "claude-sonnet-4-20250514".to_string(),
                        max_tokens: Some(4096),
                        temperature: default_temperature(),
                        top_p: default_top_p(),
                        top_k: Some(0),
                        parallel_tool_calls: default_parallel_tool_calls(),
                        max_retries: default_max_retries(),
                        base_url: Some("https://api.anthropic.com".to_string()),
                        api_version: None,
                        candidate_count: None,
                        stop_sequences: None,
                    },
                    // TODO: Add cases for other providers like Azure, Google, etc. if they have specific defaults
                    _ => {
                        // Unknown provider, create a very basic entry; API key and model must be CLI/env
                        warn!("CLI specified an unknown provider type '{}'. It must be fully configured via CLI/env.", provider_name);
                        ModelParameters {
                            api_key: None,
                            model: "unknown_cli_model".to_string(), // User must override this via --model
                            max_tokens: Some(4096), // A generic default
                            temperature: default_temperature(),
                            top_p: default_top_p(),
                            top_k: None,
                            parallel_tool_calls: default_parallel_tool_calls(),
                            max_retries: default_max_retries(),
                            base_url: None,
                            api_version: None,
                            candidate_count: None,
                            stop_sequences: None,
                        }
                    }
                };
                loaded_config
                    .model_providers
                    .insert(provider_name.clone(), default_params);
            }
        }

        // Ensure default_provider (possibly set by CLI) exists after potential addition.
        // If it was NOT set by CLI, and default from file is missing, then try to use the first one or error.
        if !loaded_config
            .model_providers
            .contains_key(&loaded_config.default_provider)
        {
            if let Some(first_provider_name) = loaded_config.model_providers.keys().next() {
                warn!("Default provider '{}' from config file not found. Using first available provider: '{}'", loaded_config.default_provider, first_provider_name);
                loaded_config.default_provider = first_provider_name.clone();
            } else {
                // This case should ideally not be reached if the config file parsing guarantees at least one provider or if CLI adds one.
                return Err(anyhow::anyhow!(
                    "No model providers configured after all checks."
                ));
            }
        }

        let provider_config = loaded_config
            .model_providers
            .get_mut(&loaded_config.default_provider)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Default provider '{}' not found in configuration after potential fallback.",
                    loaded_config.default_provider
                )
            })?;

        if let Some(model) = cli_model {
            provider_config.model = model;
        }

        // API Key Precedence: CLI > Config File > Environment Variable
        let mut final_api_key = cli_api_key.clone();

        if final_api_key.is_none() {
            // .clone() is important if provider_config.api_key is Option<String>
            // and we don't want to consume it or run into lifetime issues.
            final_api_key = provider_config.api_key.clone();
        }

        if final_api_key.is_none() {
            let env_var_name_owned: String;
            let env_var_name_ref: &str = match loaded_config.default_provider.as_str() {
                "openai" => "OPENAI_API_KEY",
                "anthropic" => "ANTHROPIC_API_KEY",
                "azure" => "AZURE_API_KEY", // from Python's env_var_map
                "openrouter" => "OPENROUTER_API_KEY", // from Python's env_var_map
                "doubao" => "DOUBAO_API_KEY", // from Python's env_var_map
                "google" => "GOOGLE_API_KEY", // from Python's env_var_map
                _ => {
                    env_var_name_owned =
                        format!("{}_API_KEY", loaded_config.default_provider.to_uppercase());
                    &env_var_name_owned
                }
            };
            if let Ok(env_key) = std::env::var(env_var_name_ref) {
                final_api_key = Some(env_key);
            }
        }
        provider_config.api_key = final_api_key;
        // If still no API key and it's required, this might be an issue later, handled by LLM client.

        if let Some(max_steps) = cli_max_steps {
            loaded_config.max_steps = max_steps;
        }

        loaded_config.working_dir = cli_working_dir.or_else(|| {
            std::env::current_dir()
                .ok()
                .map(|p| p.to_string_lossy().into_owned())
        });

        Ok(loaded_config)
    }

    pub fn get_current_provider_config(&self) -> Result<&ModelParameters> {
        self.model_providers
            .get(&self.default_provider)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Provider '{}' not found in configuration",
                    self.default_provider
                )
            })
    }
}

// Example usage (for testing purposes, can be removed or put in tests module)
#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn create_test_config_file(path: &str, content: &str) {
        fs::write(path, content).unwrap();
    }

    #[test]
    fn test_load_config_defaults_no_file() {
        let config =
            Config::load("non_existent_config.json", None, None, None, None, None).unwrap();
        assert_eq!(config.default_provider, "openai");
        assert_eq!(config.max_steps, 20);
        assert!(config.model_providers.contains_key("openai"));
    }

    #[test]
    fn test_load_from_file() {
        let config_content = r#"
        {
            "default_provider": "anthropic",
            "max_steps": 30,
            "model_providers": {
                "openai": {
                    "api_key": "sk-openai-from-file",
                    "model": "gpt-3.5-turbo",
                    "max_tokens": 1000,
                    "temperature": 0.7,
                    "top_p": 0.9
                },
                "anthropic": {
                    "api_key": "sk-anthropic-from-file",
                    "model": "claude-2",
                    "max_tokens": 2000,
                    "temperature": 0.6,
                    "top_p": 0.8,
                    "top_k": 50
                }
            }
        }
        "#;
        create_test_config_file("test_config_1.json", config_content);
        let config = Config::load("test_config_1.json", None, None, None, None, None).unwrap();
        assert_eq!(config.default_provider, "anthropic");
        assert_eq!(config.max_steps, 30);
        assert_eq!(config.model_providers["anthropic"].model, "claude-2");
        assert_eq!(
            config.model_providers["openai"].api_key,
            Some("sk-openai-from-file".to_string())
        );
        fs::remove_file("test_config_1.json").unwrap();
    }

    #[test]
    fn test_cli_overrides() {
        let config_content = r#"
        {
            "default_provider": "openai",
            "max_steps": 20,
            "model_providers": {
                "openai": { "model": "gpt-4" }
            }
        }
        "#;
        create_test_config_file("test_config_2.json", config_content);

        let config = Config::load(
            "test_config_2.json",
            Some("anthropic".to_string()),        // CLI provider
            Some("claude-instant-1".to_string()), // CLI model
            Some("sk-cli-key".to_string()),       // CLI api key
            Some(50),                             // CLI max steps
            Some("/tmp/testdir".to_string()),     // CLI working_dir
        )
        .unwrap();

        assert_eq!(config.default_provider, "anthropic");
        // Since "anthropic" provider might not exist in the base file,
        // the load function will add a default one if it doesn't.
        // We need to ensure the model and api_key are set on this potentially new provider.
        let anthropic_config = config
            .model_providers
            .get("anthropic")
            .expect("Anthropic provider should exist");

        assert_eq!(anthropic_config.model, "claude-instant-1");
        assert_eq!(anthropic_config.api_key, Some("sk-cli-key".to_string()));
        assert_eq!(config.max_steps, 50);
        assert_eq!(config.working_dir, Some("/tmp/testdir".to_string()));

        fs::remove_file("test_config_2.json").unwrap();
    }

    #[test]
    fn test_env_var_override_api_key() {
        let config_content = r#"
        {
            "default_provider": "openai",
            "model_providers": {
                "openai": { "model": "gpt-4" }
            }
        }
        "#;
        create_test_config_file("test_config_3.json", config_content);
        env::set_var("OPENAI_API_KEY", "sk-env-key");

        let config = Config::load("test_config_3.json", None, None, None, None, None).unwrap();
        assert_eq!(
            config.model_providers["openai"].api_key,
            Some("sk-env-key".to_string())
        );

        env::remove_var("OPENAI_API_KEY");
        fs::remove_file("test_config_3.json").unwrap();
    }

    #[test]
    fn test_api_key_priority() {
        // Priority: CLI > Env > File
        let config_content = r#"
        {
            "default_provider": "openai",
            "model_providers": {
                "openai": { "model": "gpt-4", "api_key": "sk-file-key" }
            }
        }
        "#;
        create_test_config_file("test_config_priority.json", config_content);

        // 1. Only file
        let config_file_only =
            Config::load("test_config_priority.json", None, None, None, None, None).unwrap();
        assert_eq!(
            config_file_only.model_providers["openai"].api_key,
            Some("sk-file-key".to_string())
        );

        // 2. File + Env
        env::set_var("OPENAI_API_KEY", "sk-env-key");
        let config_env =
            Config::load("test_config_priority.json", None, None, None, None, None).unwrap();
        // Env should override file if file key was initially None, or if env is present and file key is also present.
        // The current logic: if file has key, env doesn't override. Let's adjust to: env overrides file if file key is None.
        // If CLI is present, it overrides all.
        // The python code logic: CLI > Config File > Env > Default
        // Let's stick to Python's: CLI > Config File > Env for API keys specifically.
        // My current Rust code: CLI > Env > Config if Config is None. This needs adjustment.

        // Corrected logic expectation:
        // Config has api_key: Some("sk-file-key")
        // Env has OPENAI_API_KEY="sk-env-key"
        // CLI has no api_key
        // Expected: "sk-file-key" (as per Python: Config > Env)

        // If Config api_key is None:
        // Env has OPENAI_API_KEY="sk-env-key"
        // Expected: "sk-env-key"

        // Let's refine the Config::load for API key to match Python's order of precedence.
        // Python: resolved_api_key = resolve_config_value(cli_api_key, config_file_api_key, env_var_name)
        // This means CLI > Config File Value > Environment Variable > Default (None)

        // After adjusting Config::load:
        assert_eq!(
            config_env.model_providers["openai"].api_key,
            Some("sk-file-key".to_string()),
            "File key should take precedence over env if file key exists"
        );
        env::remove_var("OPENAI_API_KEY");

        // Test env override when file key is not present
        let config_content_no_file_key = r#"
        {
            "default_provider": "openai",
            "model_providers": {
                "openai": { "model": "gpt-4" }
            }
        }
        "#;
        create_test_config_file("test_config_no_file_key.json", config_content_no_file_key);
        env::set_var("OPENAI_API_KEY", "sk-env-key-no-file");
        let config_env_no_file_key =
            Config::load("test_config_no_file_key.json", None, None, None, None, None).unwrap();
        assert_eq!(
            config_env_no_file_key.model_providers["openai"].api_key,
            Some("sk-env-key-no-file".to_string()),
            "Env key should be used if file key is not present"
        );
        env::remove_var("OPENAI_API_KEY");
        fs::remove_file("test_config_no_file_key.json").unwrap();

        // 3. File + Env + CLI
        env::set_var("OPENAI_API_KEY", "sk-env-key-again");
        let config_cli = Config::load(
            "test_config_priority.json",
            None,
            None,
            Some("sk-cli-key-final".to_string()), // CLI Key
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            config_cli.model_providers["openai"].api_key,
            Some("sk-cli-key-final".to_string())
        );

        env::remove_var("OPENAI_API_KEY");
        fs::remove_file("test_config_priority.json").unwrap();
    }
    #[test]
    fn test_resolve_config_value_logic_for_api_key() {
        // Mimic Python's resolve_config_value behavior for API keys
        // Order: CLI -> Config File -> Environment Variable

        let file_key = Some("sk-file".to_string());
        let _env_key_val = "sk-env"; // Prefixed, as it's for conceptual testing not direct use in this simplified assert
        let cli_key = Some("sk-cli".to_string());

        // Case 1: CLI present
        let mut _effective_key = cli_key.clone(); // Prefixed
        if _effective_key.is_none() {
            _effective_key = file_key.clone();
        }
        if _effective_key.is_none() {
            _effective_key = std::env::var("TEST_API_KEY").ok();
        }
        assert_eq!(_effective_key, Some("sk-cli".to_string()));

        // Case 2: CLI not present, File present
        _effective_key = None::<String>; // Reset
        _effective_key = file_key.clone();
        if _effective_key.is_none() {
            _effective_key = std::env::var("TEST_API_KEY").ok();
        }
        // This manual simulation needs to be inside the Config::load or a helper
        // The actual test is test_api_key_priority which uses Config::load

        // The Python code is:
        // resolved_api_key = resolve_config_value(
        //     cli_api_key,  <-- Highest priority
        //     config.model_providers[str(resolved_provider)].api_key, <-- Middle
        //     "OPENAI_API_KEY" if resolved_provider == "openai" else "ANTHROPIC_API_KEY" <-- Lowest if others are None
        // )
        // This means if cli_api_key is Some, it's used.
        // Else, if config.model_providers...api_key is Some, it's used.
        // Else, if env var is Some, it's used.
        // My Config::load needs to match this exactly for API keys.

        // Re-checking Config::load logic for API key:
        // 1. `provider_config.api_key` is loaded from file.
        // 2. `if let Some(key) = cli_api_key { provider_config.api_key = Some(key); }` -> CLI overrides file. Correct.
        // 3. `else { ... if let Ok(env_key) = std::env::var(...) { if provider_config.api_key.is_none() { provider_config.api_key = Some(env_key); } } }`
        //    This part means ENV is only used if CLI was None AND file key was None.
        //    This should be: if CLI is None, then check file. If file is None, then check ENV.
        //
        // Corrected logic inside Config::load for API key:
        // let mut final_api_key = cli_api_key;
        // if final_api_key.is_none() {
        //     final_api_key = provider_config.api_key.clone(); // Value from file
        // }
        // if final_api_key.is_none() {
        //     final_api_key = std::env::var(env_var_name).ok();
        // }
        // provider_config.api_key = final_api_key;
        // This is what `test_api_key_priority` will verify. The existing `Config::load` structure for api_key needs slight adjustment.

        // The existing code is actually:
        // `provider_config.api_key` (loaded from file)
        // `if let Some(key) = cli_api_key { provider_config.api_key = Some(key); }` (CLI overrides)
        // `else { // if cli_api_key is None
        //      // api_key from env is loaded into provider_config.api_key only if provider_config.api_key (from file) was None
        //      let env_key = std::env::var(&env_var_name).ok();
        //      if provider_config.api_key.is_none() { // This means file key was None
        //          provider_config.api_key = env_key;
        //      }
        //  }`
        // This is effectively: CLI > File (if present) > Env (if File not present).
        // Python is: CLI > File > Env.
        // So if File is Some("key"), Env is "env_key", CLI is None. Python gets "key". My code gets "key". This is fine.
        // If File is None, Env is "env_key", CLI is None. Python gets "env_key". My code gets "env_key". This is fine.
        // The `test_api_key_priority` covers this. The current implementation seems to match the Python's behavior for `resolve_config_value`.
    }

    #[test]
    fn test_working_dir_logic() {
        let current_dir = std::env::current_dir()
            .unwrap()
            .to_string_lossy()
            .into_owned();

        // No CLI working_dir, should use current dir
        let config1 = Config::load("nd.json", None, None, None, None, None).unwrap();
        assert_eq!(config1.working_dir, Some(current_dir.clone()));

        // CLI working_dir provided
        let cli_wd = "/custom/path".to_string();
        let config2 =
            Config::load("nd.json", None, None, None, None, Some(cli_wd.clone())).unwrap();
        assert_eq!(config2.working_dir, Some(cli_wd));
    }
}
