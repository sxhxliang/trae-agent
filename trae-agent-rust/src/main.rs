//! # Trae Rust Agent
//!
//! This is the main entry point for the Trae Rust Agent, an LLM-based agent
//! for general purpose software engineering tasks. It provides a CLI interface
//! to interact with the agent.

mod agent;
mod cli;
mod config;
mod llm;
mod tools;
mod utils; // Add this line

use clap::Parser;
use cli::{Cli, Commands};
use tracing_subscriber::{fmt, EnvFilter};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing subscriber
    // You can control the log level using the RUST_LOG environment variable.
    // For example: RUST_LOG=trae_rust_agent=debug,info
    // This sets the default level to info, and debug for the trae_rust_agent crate.
    fmt::Subscriber::builder()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr) // Log to stderr
        .init();

    let cli_args = Cli::parse();

    match cli_args.command {
        Commands::Run(args) => {
            if let Err(e) = cli::handle_run(args).await {
                eprintln!("Error running task: {:?}", e);
                std::process::exit(1);
            }
        }
        Commands::Interactive(args) => {
            if let Err(e) = cli::handle_interactive(args).await {
                eprintln!("Error in interactive session: {:?}", e);
                std::process::exit(1);
            }
        }
        Commands::ShowConfig(args) => {
            if let Err(e) = cli::handle_show_config(args).await {
                eprintln!("Error showing config: {:?}", e);
                std::process::exit(1);
            }
        }
    }

    Ok(())
}
