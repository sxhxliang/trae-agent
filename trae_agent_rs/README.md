# Trae Agent (Rust Version)

[![Rust CI](https://github.com/placeholder/trae_rust_agent/actions/workflows/rust.yml/badge.svg)](https://github.com/placeholder/trae_rust_agent/actions/workflows/rust.yml) <!-- Placeholder badge -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Status: Alpha](https://img.shields.io/badge/Status-Alpha-red.svg)

This is a Rust implementation of the Trae Agent, an LLM-based agent for general purpose software engineering tasks.

## ✨ Features (Rust Version - Current)

*   **Core Agent Logic**: Task execution loop, LLM interaction, tool usage.
*   **CLI Interface**:
    *   `run`: Execute a task with specified parameters.
    *   `show-config`: Display current configuration.
    *   `interactive`: Start an interactive session with the agent.
*   **Configuration**: Load settings from `trae_config.json`, environment variables, and CLI arguments.
*   **LLM Support**:
    *   OpenAI client implemented and tested (via mocks).
    *   Anthropic client stubbed.
*   **Tools**:
    *   `BashTool`: Execute shell commands.
    *   `EditTool`: View, create, and edit files (str_replace, insert).
    *   `TaskDoneTool`: Allow agent to signal task completion.
    *   `SequentialThinkingTool`: For structured thought output from LLM.
*   **Patch Validation**: Agent can validate if `must_patch` is true and a non-empty patch was generated.
*   **Lakeview Summaries**: Optional LLM-based summary of agent execution.
*   **Logging**: Uses the `tracing` crate for structured logging.

## 🚀 Quick Start

### Prerequisites

*   Rust toolchain (latest stable recommended). Install via [rustup](https://rustup.rs/).
*   (Optional) Git for patch validation features.

### Build

```bash
git clone <repository-url> # Or your fork/local copy
cd trae_rust_agent
cargo build --release
```
The executable will be located at `target/release/trae_rust_agent`.

### Configuration

Create a `trae_config.json` file in the directory where you run the agent (or specify with `--config-file`). Example:

```json
{
  "default_provider": "openai",
  "max_steps": 20,
  "model_providers": {
    "openai": {
      "api_key": "YOUR_OPENAI_API_KEY",
      "model": "gpt-4o",
      "max_tokens": 8000,
      "temperature": 0.5,
      "top_p": 1.0,
      "parallel_tool_calls": true
    },
    "anthropic": {
      "api_key": "YOUR_ANTHROPIC_API_KEY",
      "model": "claude-sonnet-4",
      "max_tokens": 4096,
      "temperature": 0.5,
      "top_p": 1.0,
      "top_k": 0,
      "parallel_tool_calls": true
    }
  },
  "enable_lakeview": true,
  "lakeview_config": {
    "model_provider": "openai",
    "model_name": "gpt-3.5-turbo"
  }
}
```

Replace `YOUR_OPENAI_API_KEY` and `YOUR_ANTHROPIC_API_KEY` with your actual API keys. Alternatively, set them as environment variables:
`export OPENAI_API_KEY="your-key"`
`export ANTHROPIC_API_KEY="your-key"`

### Basic Usage

**Run a task:**
```bash
./target/release/trae_rust_agent run "Create a hello world Python script in /tmp/hello.py" --working-dir /tmp
```

**Show configuration:**
```bash
./target/release/trae_rust_agent show-config
```

**Start an interactive session:**
```bash
./target/release/trae_rust_agent interactive
```
You can then type tasks directly. Special commands: `config`, `clear_history`, `exit`.

## 🛠️ Available Tools

*   **`bash`**: Execute shell commands.
    *   Params: `command` (string, required), `timeout` (integer, optional), `working_directory` (string, optional).
*   **`str_replace_based_edit_tool`**: View, create, and edit files.
    *   Sub-commands: `view`, `create`, `str_replace`, `insert`.
    *   See tool description via LLM or code for detailed parameters.
*   **`task_done`**: Signal task completion.
    *   Params: `summary` (string, optional).
*   **`sequential_thinking`**: Record a sequence of thoughts from the LLM.
    *   Params: `thoughts` (array of strings, required).


## 🚧 Current Limitations & TODOs

*   `AnthropicClient` is stubbed and needs full implementation.
*   The `load_config <path>` command in interactive mode is a TODO.
*   Rich CLI output (like Python's `rich` console) is not implemented; `AgentEvent` system can be used for this in future.
*   Trajectory recording to JSON is available for `run` mode (`--trajectory-file`), but not explicitly for `interactive` mode sessions (though history is saved).
*   More sophisticated error handling and user feedback in CLI could be enhanced.
*   More complex agent capabilities (e.g., deeper reflection, advanced planning) from original Python agent might not be fully ported or optimized.

## 🤝 Contributing

Contributions are welcome! Please follow standard Rust development practices. (Further details can be added here).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details (assuming it's in parent directory).
