# Trae Agent Rust - Full Changelog

## [Unreleased] - 2025-07-10

### Added
- **Trajectory Recording:**
    - Created `utils/trajectory_recorder.rs` to log agent interactions.
    - Defined `TrajectoryHeader`, `Trajectory`, and `TrajectoryRecorder` structs.
    - Implemented `start_recording`, `record_agent_step`, and `finalize_recording` methods.
    - Integrated `TrajectoryRecorder` into `BaseAgent` and `TraeAgent`.
    - Updated `LLMClient` trait and implementations (`OpenAIClient`, `AnthropicClient`) with `get_provider_name()` for recorder.
    - Updated `cli.rs` to handle `--trajectory-file` option and pass it to `TraeAgent`.
- **JSON Edit Tool (`json_edit_tool.rs`):**
    - Added `jsonpath_lib` dependency (already present, verified).
    - Implemented the `view` operation for `JsonEditTool` to inspect JSON files using JSONPath expressions (already present).
    - Implemented `set`, `add` (as set), and `remove` (replaces with null) operations in `JsonEditTool` using `jsonpath_lib`.
    - Added comprehensive unit tests for `view`, `set`, `add`, and `remove` operations, including argument validation and path not found scenarios.
    - Uncommented and refined `load_json_file` helper method.
    - Registered `JsonEditTool` with the `ToolRegistry` (already present).
- **`tools` CLI Subcommand:**
    - Added a `tools` subcommand to `cli.rs` to list available tools and their descriptions.
    - Implemented `handle_tools_command` within `cli.rs` to display tool information.
- **Configuration Enhancements:**
    - Added missing fields to `ModelParameters` struct in `config.rs` (`max_retries`, `base_url`, `api_version`, `candidate_count`, `stop_sequences`) to align with Python version.
    - Updated default instantiations of `ModelParameters` to include these new fields.

### Changed
- **API Key Loading Precedence:**
    - Modified `config.rs` (`Config::load`) to align API key resolution order with Python: CLI > Config File > Environment Variable.
- **Default for `parallel_tool_calls`:**
    - Changed the default for `parallel_tool_calls` in Rust's `ModelParameters` from `true` to `false` to match Python's default.
- **Bash Tool (`bash_tool.rs`):**
    - Implemented output truncation for `stdout` and `stderr` to prevent excessively long outputs.
    - Added tests for truncation logic.
- **Edit Tool (`edit_tool.rs`):**
    - Implemented output truncation (line-based) for `view_file` and `view_dir` operations.
    - Corrected line range logic in `view_file`.
- **Tool Executor (`tools/base.rs`):**
    - Added initial scaffolding for `parallel_tool_calls` method in `ToolExecutor`.
    - Updated `BaseAgent` to use `parallel_tool_calls` or `sequential_tool_calls` based on the `parallel_tool_calls` configuration.

### Fixed
- Numerous compilation errors and test failures during the implementation of the above features.
- Corrected `jsonpath_lib` usage in `JsonEditTool` after several attempts.
- Addressed module visibility and path issues for CLI handlers.
- Fixed test failures in `BashTool` related to output length assertions.
- Corrected private function access for `default_max_retries` in `config.rs` tests.
- Resolved duplicate method definition for `parallel_tool_calls`.
- Fixed various import and type issues across multiple files.

### Removed
- Unused `record_llm_interaction` method from `trajectory_recorder.rs`.
- Redundant trajectory saving logic from `cli.rs` as it's now handled by `TrajectoryRecorder`.
- Temporarily commented out `load_json_file` in `JsonEditTool` as it was unused after refactoring `view` op (will be needed for mutations).
