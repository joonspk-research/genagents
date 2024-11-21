from pathlib import Path

# API Keys
OPENAI_API_KEY = "API_KEY"  # Replace with your actual OpenAI API key
ANTHROPIC_API_KEY = "API_KEY"  # Replace with your actual Anthropic API key

# Owner Information
KEY_OWNER = "NAME"  # Replace with the name of the key owner

# Debugging Configuration
DEBUG = False  # Set to True for enabling debug logs

# Configuration for Chunk Size
MAX_CHUNK_SIZE = 4  # Maximum size of data chunks to process

# LLM Configuration
LLM_VERS = "claude-3-5-sonnet-20241022"  
# Options: 
# - "gpt-4o-mini" (OpenAI GPT model)
# - "claude-3-5-sonnet-20241022" (Anthropic Claude model)
# - "gpt4all" (Open-source GPT model)

# GPT4All Model Settings
LLM_MODEL = "MODEL GPT4ALL"  
# Options: 
# - "orca-mini-3b-gguf2-q4_0.gguf" (3 Billion Parameters, 4GB RAM)
# - "Meta-Llama-3-8B-Instruct.Q4_0.gguf" (8 Billion Parameters, 8GB RAM)
# - "Nous-Hermes-2-Mistral-7B-DPO.Q4_0.gguf" (7 Billion Parameters, 8GB RAM)

# Notes:
# - Choose the model based on your hardware capabilities and task requirements.
# - Ensure you have sufficient RAM to load the selected model.
# - Visit 📖 [GPT4All Documentation](https://docs.gpt4all.io/gpt4all_python/home.html) for detailed information.

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory Configurations
# - Populations Directory: Used for managing agent populations
# - Prompt Template Directory: Contains LLM prompt templates
POPULATIONS_DIR = BASE_DIR / "agent_bank" / "populations"
LLM_PROMPT_DIR = BASE_DIR / "simulation_engine" / "prompt_template"

# Note:
# - Ensure `POPULATIONS_DIR` and `LLM_PROMPT_DIR` exist in your project structure.
# - Adjust the paths as needed for your specific setup.
