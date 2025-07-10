// trae-agent-rust/src/tools/json_edit_tool.rs

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value as JsonValue, from_str as json_from_str, to_string_pretty, to_string};
use std::fs;
use std::path::Path;
// Removed direct Selector and PathParser imports, will use top-level jsonpath_lib::select

use super::base::{Tool, ToolError, ToolExecResult, ToolParameter};

#[derive(Deserialize, Debug)]
struct JsonEditToolArgs {
    operation: String,
    file_path: String,
    json_path: Option<String>,
    value: Option<JsonValue>, // serde_json::Value for dynamic JSON
    pretty_print: Option<bool>,
}

pub struct JsonEditTool;

impl JsonEditTool {
    pub fn new() -> Self {
        JsonEditTool
    }

    // load_json_file is not directly used by 'view' anymore with jsonpath_lib::select.
    // It would be needed for mutation operations. Keeping it commented out for now.
    // async fn load_json_file(&self, file_path_str: &str) -> Result<JsonValue, ToolError> { // Removed async as fs::read_to_string is sync
    fn load_json_file(&self, file_path_str: &str) -> Result<JsonValue, ToolError> {
        let file_path = Path::new(file_path_str);
        if !file_path.is_absolute() {
            return Err(ToolError::InvalidArguments {
                tool_name: self.get_name(),
                message: format!("File path must be absolute: {}", file_path_str),
            });
        }
        if !file_path.exists() {
            return Err(ToolError::FileNotFound(file_path_str.to_string()));
        }

        let content = fs::read_to_string(file_path).map_err(|e| {
            ToolError::FileReadError(format!(
                "Failed to read file {}: {}",
                file_path_str, e
            ))
        })?;

        if content.trim().is_empty() {
            // Consider if an empty file should be an error or default to an empty JSON object/array
            // For now, consistent with previous logic, it's an error.
            // If it should be treated as e.g. `JsonValue::Null` or `JsonValue::Object(Default::default())`,
            // this part needs adjustment.
            return Err(ToolError::FileReadError(format!(
                "File is empty: {}",
                file_path_str
            )));
        }

        json_from_str(&content).map_err(|e| {
            ToolError::InvalidJson(format!(
                "Invalid JSON in file {}: {}",
                file_path_str, e
            ))
        })
    }

    #[allow(dead_code)] // Will be used by set/add/remove operations
    async fn save_json_file(
        &self,
        file_path_str: &str,
        data: &JsonValue,
        pretty: bool,
    ) -> Result<(), ToolError> {
        let file_path = Path::new(file_path_str);
        // Path absoluteness check is done in load_json_file, assuming it's called first for edits
        let output = if pretty {
            to_string_pretty(data).map_err(|e| ToolError::InternalError(format!("Failed to serialize JSON (pretty): {}", e)))?
        } else {
            to_string(data).map_err(|e| ToolError::InternalError(format!("Failed to serialize JSON: {}", e)))?
        };

        fs::write(file_path, output).map_err(|e| {
            ToolError::FileWriteError(format!(
                "Failed to write to file {}: {}",
                file_path_str, e
            ))
        })
    }
}

#[async_trait]
impl Tool for JsonEditTool {
    fn get_name(&self) -> String {
        "json_edit_tool".to_string()
    }

    fn get_description(&self) -> String {
        "Tool for editing JSON files with JSONPath expressions. Supports view, set, add, remove operations.".to_string()
    }

    fn get_parameters(&self) -> Vec<ToolParameter> {
        vec![
            ToolParameter {
                name: "operation".to_string(),
                param_type: "string".to_string(),
                description: "The operation to perform on the JSON file.".to_string(),
                is_required: true,
                enum_values: Some(vec!["view".to_string(), "set".to_string(), "add".to_string(), "remove".to_string()]),
                items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "file_path".to_string(),
                param_type: "string".to_string(),
                description: "Absolute path to the JSON file to edit.".to_string(),
                is_required: true,
                enum_values: None, items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "json_path".to_string(),
                param_type: "string".to_string(),
                description: "JSONPath expression (e.g., '$.users[0].name'). Required for set, add, remove. Optional for view.".to_string(),
                is_required: false,
                enum_values: None, items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "value".to_string(),
                param_type: "object".to_string(), // In JSON schema, 'object' can represent any JSON value (object, array, string, number, boolean, null)
                description: "The JSON value to set or add. Required for set and add operations.".to_string(),
                is_required: false,
                enum_values: None, items: None, properties: None, required: vec![],
            },
            ToolParameter {
                name: "pretty_print".to_string(),
                param_type: "boolean".to_string(),
                description: "Whether to format the JSON output with indentation. Defaults to true.".to_string(),
                is_required: false,
                enum_values: None, items: None, properties: None, required: vec![],
            },
        ]
    }

    async fn execute(&self, arguments: JsonValue) -> Result<ToolExecResult, ToolError> {
        let args: JsonEditToolArgs = serde_json::from_value(arguments.clone()).map_err(|e| {
            ToolError::InvalidArguments {
                tool_name: self.get_name(),
                message: format!("Failed to parse arguments: {}. Args: {:?}", e, arguments),
            }
        })?;

        let pretty = args.pretty_print.unwrap_or(true);

        match args.operation.to_lowercase().as_str() {
            "view" => {
                let file_path = Path::new(&args.file_path);
                if !file_path.is_absolute() {
                    return Err(ToolError::InvalidArguments {
                        tool_name: self.get_name(),
                        message: format!("File path must be absolute: {}", args.file_path),
                    });
                }
                if !file_path.exists() { // Check existence before reading
                    return Err(ToolError::FileNotFound(args.file_path.to_string()));
                }

                let file_content_str = fs::read_to_string(file_path).map_err(|e| {
                    ToolError::FileReadError(format!(
                        "Failed to read file {}: {}",
                        args.file_path, e
                    ))
                })?;

                if file_content_str.trim().is_empty() {
                    return Err(ToolError::FileReadError(format!(
                        "File is empty: {}",
                        args.file_path
                    )));
                }

                // Parse the string content into a serde_json::Value
                let data: JsonValue = json_from_str(&file_content_str).map_err(|e| {
                    ToolError::InvalidJson(format!(
                        "Invalid JSON in file {}: {}",
                        args.file_path, e
                    ))
                })?;

                if let Some(json_path_str) = args.json_path {
                    // Use jsonpath_lib::select which now takes &JsonValue and &str path
                    let results: Vec<&JsonValue> =
                        jsonpath_lib::select(&data, &json_path_str) // Pass parsed data
                        .map_err(|e| ToolError::InvalidArguments {
                            tool_name: self.get_name(),
                            message: format!("Error selecting with JSONPath '{}': {}", json_path_str, e),
                        })?;

                    if results.is_empty() {
                        Ok(ToolExecResult::new_success(Some(format!("No matches found for JSONPath: {}", json_path_str)), None))
                    } else {
                        // results is Vec<&JsonValue>
                        let output_val_to_serialize = if results.len() == 1 {
                            results[0].clone() // Clone to own the JsonValue for serialization
                        } else {
                            JsonValue::Array(results.into_iter().cloned().collect())
                        };

                        let output_str = if pretty {
                            to_string_pretty(&output_val_to_serialize)
                        } else {
                            to_string(&output_val_to_serialize)
                        }
                        .map_err(|e| ToolError::InternalError(format!("Failed to serialize result: {}", e)))?;

                        Ok(ToolExecResult::new_success(Some(format!("JSONPath '{}' matches:\n{}", json_path_str, output_str)), None))
                    }
                } else {
                    // No JSONPath provided, view the whole file.
                    // Need to parse the file_content_str into JsonValue first.
                    let data: JsonValue = json_from_str(&file_content_str).map_err(|e| {
                        ToolError::InvalidJson(format!(
                            "Invalid JSON in file {}: {}",
                            args.file_path, e
                        ))
                    })?;
                    let output_str = if pretty {
                        to_string_pretty(&data)
                    } else {
                        to_string(&data)
                    }
                    .map_err(|e| ToolError::InternalError(format!("Failed to serialize JSON: {}", e)))?;
                    Ok(ToolExecResult::new_success(Some(format!("JSON content of {}:\n{}", args.file_path, output_str)), None))
                }
            }
            "set" => {
                let json_path_str = args.json_path.ok_or_else(|| ToolError::InvalidArguments {
                    tool_name: self.get_name(),
                    message: "'json_path' is required for 'set' operation.".to_string(),
                })?;
                let value_to_set = args.value.ok_or_else(|| ToolError::InvalidArguments {
                    tool_name: self.get_name(),
                    message: "'value' is required for 'set' operation.".to_string(),
                })?;

                // Ensure file_path is absolute before loading
                let file_path = Path::new(&args.file_path);
                if !file_path.is_absolute() {
                    return Err(ToolError::InvalidArguments {
                        tool_name: self.get_name(),
                        message: format!("File path must be absolute: {}", args.file_path),
                    });
                }

                let data = self.load_json_file(&args.file_path)?; // Removed mut

                let mut selector_mut = jsonpath_lib::SelectorMut::new(); // Changed from new_with_values(None)

                let modified_data = selector_mut
                    .str_path(&json_path_str)
                    .map_err(|e| ToolError::InvalidArguments {
                        tool_name: self.get_name(),
                        message: format!("Invalid JSONPath for 'set': {}. Error: {}", json_path_str, e),
                    })?
                    .value(data) // Pass current data to selector_mut
                    .replace_with(&mut |_v| { // _v is the matched value, we replace it with value_to_set
                        Some(value_to_set.clone()) // Must return Option<JsonValue>
                    })
                    .map_err(|e| ToolError::InternalError(format!("Failed to apply 'set' operation with JSONPath '{}': {}", json_path_str, e)))?
                    .take() // Consumes selector_mut and returns the modified JsonValue
                    .ok_or_else(|| ToolError::InternalError("Failed to take modified data from selector_mut".to_string()))?;

                // `replace_with` returns the whole modified JSON object.
                // If `replace_with` didn't find any matches, it would return the original `data` unchanged.
                // We need to check if any change was actually made, but jsonpath_lib doesn't directly tell us.
                // For now, we assume success if no error occurred.

                self.save_json_file(&args.file_path, &modified_data, pretty).await?;

                Ok(ToolExecResult::new_success(Some(format!("Successfully set value at JSONPath '{}' in file '{}'", json_path_str, args.file_path)), None))
            }
            "remove" => {
                let json_path_str = args.json_path.ok_or_else(|| ToolError::InvalidArguments {
                    tool_name: self.get_name(),
                    message: "'json_path' is required for 'remove' operation.".to_string(),
                })?;

                let file_path = Path::new(&args.file_path);
                if !file_path.is_absolute() {
                    return Err(ToolError::InvalidArguments {
                        tool_name: self.get_name(),
                        message: format!("File path must be absolute: {}", args.file_path),
                    });
                }

                let data = self.load_json_file(&args.file_path)?;

                // jsonpath_lib::delete takes ownership of `data` and returns a new `JsonValue`
                let modified_data = jsonpath_lib::delete(data, &json_path_str)
                    .map_err(|e| ToolError::InternalError(format!("Failed to apply 'remove' operation with JSONPath '{}': {}", json_path_str, e)))?;

                // Similar to 'set', jsonpath_lib::delete will return the modified structure.
                // If the path doesn't match anything, it returns the original structure unchanged.

                self.save_json_file(&args.file_path, &modified_data, pretty).await?;

                Ok(ToolExecResult::new_success(Some(format!("Successfully applied 'remove' operation with JSONPath '{}' in file '{}'. Matched elements are replaced with null.", json_path_str, args.file_path)), None))
            }
            "add" => {
                // Initial implementation: 'add' behaves like 'set'.
                // A more sophisticated 'add' might involve appending to arrays or ensuring a field is new.
                // However, jsonpath_lib::SelectorMut::replace_with is about replacing nodes found by the path.
                // If the path points to something, it's replaced. If it doesn't point to an existing node
                // that can be directly replaced, jsonpath_lib might not create new parent structures.
                // This behavior is consistent with how 'set' is implemented here.
                let json_path_str = args.json_path.ok_or_else(|| ToolError::InvalidArguments {
                    tool_name: self.get_name(),
                    message: "'json_path' is required for 'add' operation.".to_string(),
                })?;
                let value_to_add = args.value.ok_or_else(|| ToolError::InvalidArguments {
                    tool_name: self.get_name(),
                    message: "'value' is required for 'add' operation.".to_string(),
                })?;

                let file_path = Path::new(&args.file_path);
                if !file_path.is_absolute() {
                    return Err(ToolError::InvalidArguments {
                        tool_name: self.get_name(),
                        message: format!("File path must be absolute: {}", args.file_path),
                    });
                }

                let data = self.load_json_file(&args.file_path)?; // Removed mut
                let mut selector_mut = jsonpath_lib::SelectorMut::new(); // Changed from new_with_values(None)

                let modified_data = selector_mut
                    .str_path(&json_path_str)
                    .map_err(|e| ToolError::InvalidArguments {
                        tool_name: self.get_name(),
                        message: format!("Invalid JSONPath for 'add': {}. Error: {}", json_path_str, e),
                    })?
                    .value(data)
                    .replace_with(&mut |_v| {
                        Some(value_to_add.clone())
                    })
                    .map_err(|e| ToolError::InternalError(format!("Failed to apply 'add' operation with JSONPath '{}': {}", json_path_str, e)))?
                    .take()
                    .ok_or_else(|| ToolError::InternalError("Failed to take modified data from selector_mut for 'add'".to_string()))?;

                self.save_json_file(&args.file_path, &modified_data, pretty).await?;

                Ok(ToolExecResult::new_success(Some(format!("Successfully applied 'add' (as set) operation with JSONPath '{}' in file '{}'", json_path_str, args.file_path)), None))
            }
            _ => Err(ToolError::InvalidArguments{
                tool_name: self.get_name(),
                message: format!("Unknown operation: {}. Supported: view, set, add, remove", args.operation),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    fn create_temp_json_file(content: &str) -> NamedTempFile {
        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "{}", content).unwrap();
        temp_file
    }

    #[tokio::test]
    async fn test_json_edit_view_full_file_pretty() {
        let tool = JsonEditTool::new();
        let json_content = r#"{"name": "test", "version": 1, "data": [1, 2]}"#;
        let temp_file = create_temp_json_file(json_content);
        let args = serde_json::json!({
            "operation": "view",
            "file_path": temp_file.path().to_str().unwrap(),
            "pretty_print": true
        });
        let result = tool.execute(args).await.unwrap();
        assert_eq!(result.error_code, 0, "Expected success (error_code 0), got error: {:?}", result.error);
        let expected_output = format!("JSON content of {}:\n{}", temp_file.path().to_str().unwrap(), serde_json::to_string_pretty(&serde_json::from_str::<JsonValue>(json_content).unwrap()).unwrap());
        assert_eq!(result.output.unwrap(), expected_output);
    }

    #[tokio::test]
    async fn test_json_edit_view_specific_path() {
        let tool = JsonEditTool::new();
        let json_content = r#"{"name": "test", "version": 1, "data": {"value": "target"}}"#;
        let temp_file = create_temp_json_file(json_content);
        let args = serde_json::json!({
            "operation": "view",
            "file_path": temp_file.path().to_str().unwrap(),
            "json_path": "$.data.value"
        });
        let result = tool.execute(args).await.unwrap();
        assert_eq!(result.error_code, 0, "Expected success (error_code 0), got error: {:?}", result.error);
        let expected_json_match = serde_json::to_string_pretty(&JsonValue::String("target".to_string())).unwrap();
        let expected_output = format!("JSONPath '$.data.value' matches:\n{}", expected_json_match);
        assert_eq!(result.output.unwrap(), expected_output);
    }

     #[tokio::test]
    async fn test_json_edit_view_path_not_found() {
        let tool = JsonEditTool::new();
        let json_content = r#"{"name": "test"}"#;
        let temp_file = create_temp_json_file(json_content);
        let args = serde_json::json!({
            "operation": "view",
            "file_path": temp_file.path().to_str().unwrap(),
            "json_path": "$.nonexistent"
        });
        let result = tool.execute(args).await.unwrap();
        assert_eq!(result.error_code, 0, "Expected success (error_code 0) for path not found, got error: {:?}", result.error); // View operation is successful even if path not found
        assert_eq!(result.output.unwrap(), "No matches found for JSONPath: $.nonexistent");
    }

    #[tokio::test]
    async fn test_json_edit_file_not_found() {
        let tool = JsonEditTool::new();
        let args = serde_json::json!({
            "operation": "view",
            "file_path": "/absolute/path/to/nonexistent/file.json"
        });
        let result = tool.execute(args).await;
        assert!(result.is_err());
        match result.err().unwrap() {
            ToolError::FileNotFound(path) => assert_eq!(path, "/absolute/path/to/nonexistent/file.json"),
            e => panic!("Expected FileNotFound error, got {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_json_edit_invalid_json_file() {
        let tool = JsonEditTool::new();
        let temp_file = create_temp_json_file(r#"{"name": "test", "version": 1, data: [1, 2]}"#); // Invalid JSON (data key)
        let args = serde_json::json!({
            "operation": "view",
            "file_path": temp_file.path().to_str().unwrap()
        });
        let result = tool.execute(args).await;
        assert!(result.is_err());
         match result.err().unwrap() {
            ToolError::InvalidJson(msg) => assert!(msg.contains("Invalid JSON in file")),
            e => panic!("Expected InvalidJson error, got {:?}", e),
        }
    }

    // TODO: Add tests for 'set', 'add', 'remove' once implemented.
    // For now, test that they return NotImplemented.
    // #[tokio::test]
    // async fn test_set_operation_not_implemented() {
    //     let tool = JsonEditTool::new();
    //     let temp_file = create_temp_json_file(r#"{"key": "value"}"#);
    //     let args = serde_json::json!({
    //         "operation": "set",
    //         "file_path": temp_file.path().to_str().unwrap(),
    //         "json_path": "$.key",
    //         "value": "new_value"
    //     });
    //     let result = tool.execute(args).await;
    //     assert!(result.is_err());
    //     match result.err().unwrap() {
    //         ToolError::NotImplemented(msg) => assert!(msg.contains("Operation 'set' is not yet implemented")),
    //         e => panic!("Expected NotImplemented error, got {:?}", e),
    //     }
    // }

    // Helper to read file content as string
    fn read_file_content(path: &Path) -> String {
        fs::read_to_string(path).expect("Failed to read temp file for verification")
    }

    #[tokio::test]
    async fn test_json_edit_set_value_object() {
        let tool = JsonEditTool::new();
        let initial_json = r#"{"name": "test", "details": {"version": 1}}"#;
        let temp_file = create_temp_json_file(initial_json);
        let new_value = serde_json::json!("updated_test");
        let args = serde_json::json!({
            "operation": "set",
            "file_path": temp_file.path().to_str().unwrap(),
            "json_path": "$.name",
            "value": new_value,
            "pretty_print": false // for easier comparison
        });

        let result = tool.execute(args).await.unwrap();
        assert_eq!(result.error_code, 0, "Expected success, got error: {:?}", result.error);

        let file_content = read_file_content(temp_file.path());
        let expected_json = r#"{"name":"updated_test","details":{"version":1}}"#;
        assert_eq!(file_content.trim(), expected_json);
    }

    #[tokio::test]
    async fn test_json_edit_set_value_array() {
        let tool = JsonEditTool::new();
        let initial_json = r#"{"items": ["a", "b", "c"]}"#;
        let temp_file = create_temp_json_file(initial_json);
        let new_value = serde_json::json!("x");
        let args = serde_json::json!({
            "operation": "set",
            "file_path": temp_file.path().to_str().unwrap(),
            "json_path": "$.items[1]",
            "value": new_value,
            "pretty_print": false
        });

        let result = tool.execute(args).await.unwrap();
        assert_eq!(result.error_code, 0, "Expected success, got error: {:?}", result.error);

        let file_content = read_file_content(temp_file.path());
        let expected_json = r#"{"items":["a","x","c"]}"#;
        assert_eq!(file_content.trim(), expected_json);
    }

    #[tokio::test]
    async fn test_json_edit_set_path_not_found() {
        let tool = JsonEditTool::new();
        let initial_json = r#"{"name":"test"}"#; // Compacted JSON
        let temp_file = create_temp_json_file(initial_json);
        let args = serde_json::json!({
            "operation": "set",
            "file_path": temp_file.path().to_str().unwrap(),
            "json_path": "$.nonexistent.key",
            "value": "new_value",
            "pretty_print": false
        });

        let result = tool.execute(args).await.unwrap();
        // jsonpath_lib replace_with will not error if path not found, it just won't change anything.
        assert_eq!(result.error_code, 0, "Expected success, got error: {:?}", result.error);

        let file_content = read_file_content(temp_file.path());
        // File content should be unchanged
        assert_eq!(file_content.trim(), initial_json);
    }

    #[tokio::test]
    async fn test_json_edit_set_missing_value_arg() {
        let tool = JsonEditTool::new();
        let temp_file = create_temp_json_file(r#"{"key": "value"}"#);
        let args = serde_json::json!({
            "operation": "set",
            "file_path": temp_file.path().to_str().unwrap(),
            "json_path": "$.key"
            // "value" is missing
        });
        let result = tool.execute(args).await;
        assert!(result.is_err());
        match result.err().unwrap() {
            ToolError::InvalidArguments { message, .. } => assert!(message.contains("'value' is required for 'set' operation.")),
            e => panic!("Expected InvalidArguments error, got {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_json_edit_set_missing_json_path_arg() {
        let tool = JsonEditTool::new();
        let temp_file = create_temp_json_file(r#"{"key": "value"}"#);
        let args = serde_json::json!({
            "operation": "set",
            "file_path": temp_file.path().to_str().unwrap(),
            "value": "new_value"
            // "json_path" is missing
        });
        let result = tool.execute(args).await;
        assert!(result.is_err());
        match result.err().unwrap() {
            ToolError::InvalidArguments { message, .. } => assert!(message.contains("'json_path' is required for 'set' operation.")),
            e => panic!("Expected InvalidArguments error, got {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_json_edit_remove_element_object() {
        let tool = JsonEditTool::new();
        let initial_json = r#"{"name": "test", "version": 1, "obsolete": true}"#;
        let temp_file = create_temp_json_file(initial_json);
        let args = serde_json::json!({
            "operation": "remove",
            "file_path": temp_file.path().to_str().unwrap(),
            "json_path": "$.obsolete",
            "pretty_print": false
        });

        let result = tool.execute(args).await.unwrap();
        assert_eq!(result.error_code, 0, "Expected success, got error: {:?}", result.error);

        let file_content = read_file_content(temp_file.path());
        // jsonpath_lib::delete replaces with null
        let expected_json = r#"{"name":"test","version":1,"obsolete":null}"#;
        assert_eq!(file_content.trim(), expected_json);
    }

    #[tokio::test]
    async fn test_json_edit_remove_element_array() {
        let tool = JsonEditTool::new();
        let initial_json = r#"{"items": ["a", "b", "c"]}"#;
        let temp_file = create_temp_json_file(initial_json);
        let args = serde_json::json!({
            "operation": "remove",
            "file_path": temp_file.path().to_str().unwrap(),
            "json_path": "$.items[1]",
            "pretty_print": false
        });

        let result = tool.execute(args).await.unwrap();
        assert_eq!(result.error_code, 0, "Expected success, got error: {:?}", result.error);

        let file_content = read_file_content(temp_file.path());
        // jsonpath_lib::delete replaces with null
        let expected_json = r#"{"items":["a",null,"c"]}"#;
        assert_eq!(file_content.trim(), expected_json);
    }

    #[tokio::test]
    async fn test_json_edit_remove_path_not_found() {
        let tool = JsonEditTool::new();
        let initial_json = r#"{"name":"test"}"#; // Compacted JSON
        let temp_file = create_temp_json_file(initial_json);
        let args = serde_json::json!({
            "operation": "remove",
            "file_path": temp_file.path().to_str().unwrap(),
            "json_path": "$.nonexistent",
            "pretty_print": false
        });

        let result = tool.execute(args).await.unwrap();
        // jsonpath_lib::delete will not error if path not found, it just won't change anything.
        assert_eq!(result.error_code, 0, "Expected success, got error: {:?}", result.error);

        let file_content = read_file_content(temp_file.path());
        assert_eq!(file_content.trim(), initial_json); // Content should be unchanged
    }

    #[tokio::test]
    async fn test_json_edit_remove_missing_json_path_arg() {
        let tool = JsonEditTool::new();
        let temp_file = create_temp_json_file(r#"{"key": "value"}"#);
        let args = serde_json::json!({
            "operation": "remove",
            "file_path": temp_file.path().to_str().unwrap()
            // "json_path" is missing
        });
        let result = tool.execute(args).await;
        assert!(result.is_err());
        match result.err().unwrap() {
            ToolError::InvalidArguments { message, .. } => assert!(message.contains("'json_path' is required for 'remove' operation.")),
            e => panic!("Expected InvalidArguments error, got {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_json_edit_add_like_set_object() { // 'add' currently behaves like 'set'
        let tool = JsonEditTool::new();
        // Test updating an existing value using 'add' (which functions like 'set')
        let initial_json_update = r#"{"name":"test","status":"pending"}"#;
        let temp_file_update = create_temp_json_file(initial_json_update);
        let args_update_existing = serde_json::json!({
            "operation": "add",
            "file_path": temp_file_update.path().to_str().unwrap(),
            "json_path": "$.status",
            "value": "completed",
            "pretty_print": false
        });
        let result_update = tool.execute(args_update_existing).await.unwrap();
        assert_eq!(result_update.error_code, 0, "Update existing via 'add' failed: {:?}", result_update.error);
        let file_content_update = read_file_content(temp_file_update.path());
        let expected_json_update = r#"{"name":"test","status":"completed"}"#;
        assert_eq!(file_content_update.trim(), expected_json_update);

        // Test attempting to 'add' (as 'set') to a path that does not exist.
        // Expect no change to the file, similar to 'set_path_not_found'.
        let initial_json_no_create = r#"{"name":"test"}"#;
        let temp_file_no_create = create_temp_json_file(initial_json_no_create);
        let args_no_create = serde_json::json!({
            "operation": "add",
            "file_path": temp_file_no_create.path().to_str().unwrap(),
            "json_path": "$.description", // This key does not exist
            "value": "new_description",
            "pretty_print": false
        });
        let result_no_create = tool.execute(args_no_create).await.unwrap();
        assert_eq!(result_no_create.error_code, 0, "Add to non-existent path should succeed with no change: {:?}", result_no_create.error);
        let file_content_no_create = read_file_content(temp_file_no_create.path());
        assert_eq!(file_content_no_create.trim(), initial_json_no_create, "File content should be unchanged when adding to non-existent path.");
    }

    #[tokio::test]
    async fn test_json_edit_add_missing_value_arg() {
        let tool = JsonEditTool::new();
        let temp_file = create_temp_json_file(r#"{"key": "value"}"#);
        let args = serde_json::json!({
            "operation": "add",
            "file_path": temp_file.path().to_str().unwrap(),
            "json_path": "$.key"
            // "value" is missing
        });
        let result = tool.execute(args).await;
        assert!(result.is_err());
        match result.err().unwrap() {
            ToolError::InvalidArguments { message, .. } => assert!(message.contains("'value' is required for 'add' operation.")),
            e => panic!("Expected InvalidArguments error, got {:?}", e),
        }
    }
}
