# genagents: Generative Agent Simulations of 1,000 People

![Cover Image](static_dir/cover3.png)

## Overview

This project introduces a novel agent architecture that simulates the attitudes and behaviors of real individuals by applying large language models (LLMs) to qualitative interviews about their lives. These agents replicate participants' responses on various social science measures, providing a foundation for new tools to investigate individual and collective behavior.

In the coming months, the authors at Stanford University plan to make available generative agents of 1,000 people—based on 2,000 hours of interviews—via a restricted API for research purposes. To support research while protecting participant privacy, this restricted access will offer a two-pronged system:

1. **Open Access to Aggregated Responses on Fixed Tasks**: Researchers can access aggregated data to analyze general trends and patterns.
2. **Restricted Access to Individual Responses on Open Tasks**: Researchers can request access to individual agents' responses for more detailed studies, subject to a review process ensuring ethical considerations are met.

This codebase offers two main components:

1. **Codebase for Creating and Interacting with Generative Agents**: Tools to build new agents based on your own data and interact with them. Query agents with surveys, experiments, and other stimuli to study their responses.
2. **Demographic Agent Banks**: A bank of over 3,000 agents created using demographic information from the General Social Survey (GSS) as a starting point to explore the codebase. *Note: The names and addresses are fictional.*

Additionally, to provide users with a sense of the interview-based agents, we offer an example generative agent of one of the authors, created using the same interview protocol used in our paper.

## Table of Contents

- [Installation](#installation)
  - [Requirements](#requirements)
  - [Dependencies](#dependencies)
  - [Configuration](#configuration)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
  - [Creating a Generative Agent](#creating-a-generative-agent)
  - [Interacting with Agents](#interacting-with-agents)
    - [Categorical Responses](#categorical-responses)
    - [Numerical Responses](#numerical-responses)
    - [Open-Ended Questions](#open-ended-questions)
  - [Memory and Reflection](#memory-and-reflection)
    - [Adding Memories](#adding-memories)
    - [Reflection](#reflection)
  - [Saving and Loading Agents](#saving-and-loading-agents)
- [Sample Agent](#sample-agent)
- [Agent Bank Access](#agent-bank-access)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)
- [Contact](#contact)

## Installation

### Requirements

- Python 3.7 or higher
- An OpenAI API key with access to GPT-4 or GPT-3.5-turbo models

### Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### Configuration

Create a `settings.py` file in the `simulation_engine` folder (where `example-settings.py` is located). Place the following content in `settings.py`:

```python
from pathlib import Path

OPENAI_API_KEY = "YOUR_API_KEY"
KEY_OWNER = "YOUR_NAME"

DEBUG = False

MAX_CHUNK_SIZE = 4

LLM_VERS = "gpt-4o-mini"

BASE_DIR = f"{Path(__file__).resolve().parent.parent}"

POPULATIONS_DIR = f"{BASE_DIR}/agent_bank/populations"
LLM_PROMPT_DIR = f"{BASE_DIR}/simulation_engine/prompt_template"
```

Replace `"YOUR_API_KEY"` with your actual OpenAI API key and `"YOUR_NAME"` with your name.

## Repository Structure

- `genagents/`: Core module for creating and interacting with generative agents
  - `genagents.py`: Main class for the generative agent
  - `modules/`: Submodules for interaction and memory management
    - `interaction.py`: Handles agent interactions and responses
    - `memory_stream.py`: Manages the agent's memory and reflections
- `simulation_engine/`: Contains settings and global methods
  - `prompt_template/`: All LLM prompts used in this project
  - `settings.py`: Configuration settings for the simulation engine
  - `global_methods.py`: Helper functions used across modules
  - `gpt_structure.py`: Functions for interacting with the GPT models
  - `llm_json_parser.py`: Parses JSON outputs from language models
- `agent_bank/`: Directory for storing agent data
  - `populations/`: Contains pre-generated agents
    - `gss_agents/`: Demographic agent data based on the GSS
    - `single_agent/`: Example agent data (see [Sample Agent](#sample-agent))
- `README.md`: This readme file
- `requirements.txt`: List of Python dependencies

## Usage

### Creating a Generative Agent

To create a new generative agent, use the `GenerativeAgent` class from the `genagents` module:

```python
from genagents.genagents import GenerativeAgent

# Initialize a new agent
agent = GenerativeAgent()

# Update the agent's scratchpad with personal information
agent.update_scratch({
    "first_name": "John",
    "last_name": "Doe",
    "age": 30,
    "occupation": "Software Engineer",
    "interests": ["reading", "hiking", "coding"]
})
```

The `update_scratch` method allows you to add personal attributes to the agent, which are used in interactions.

### Interacting with Agents

#### Categorical Responses

You can ask the agent to respond to categorical survey questions:

```python
questions = {
    "Do you enjoy outdoor activities?": ["Yes", "No", "Sometimes"]
}

response = agent.categorical_resp(questions)
print(response["responses"])
```

#### Numerical Responses

For numerical questions:

```python
questions = {
    "On a scale of 1 to 10, how much do you enjoy coding?": [1, 10]
}

response = agent.numerical_resp(questions, float_resp=False)
print(response["responses"])
```

#### Open-Ended Questions

Have the agent generate open-ended responses:

```python
dialogue = [
    ("Interviewer", "Tell me about your favorite hobby."),
]

response = agent.utterance(dialogue)
print(response)
```

### Memory and Reflection

Agents have a memory stream that allows them to remember and reflect on experiences.

#### Adding Memories

```python
agent.remember("Went for a hike in the mountains.", time_step=1)
```

#### Reflection

Agents can reflect on their memories to form new insights:

```python
agent.reflect(anchor="outdoor activities", time_step=2)
```

### Saving and Loading Agents

You can save the agent's state to a directory for later use:

```python
agent.save("path/to/save_directory")
```

To load an existing agent:

```python
agent = GenerativeAgent(agent_folder="path/to/save_directory")
```

## Sample Agent

A sample agent is provided in the `agent_bank/populations/single_agent/` directory. This agent includes a pre-populated memory stream and scratchpad information for demonstration purposes.

You can load and interact with the sample agent as follows:

```python
agent = GenerativeAgent(agent_folder="agent_bank/populations/single_agent/01fd7d2a-0357-4c1b-9f3e-8eade2d537ae")

# Interact with the agent
questions = {
    "Do you enjoy outdoor activities?": ["Yes", "No", "Sometimes"]
}
response = agent.categorical_resp(questions)
print(response["responses"])
```

## Agent Bank Access

Due to participant privacy concerns, the full agent bank containing over 1,000 generative agents based on real interviews is not publicly available at the moment. However, we plan to make aggregated responses on fixed tasks accessible for general research use in the coming months. Researchers interested in accessing individual responses on open tasks can request restricted access by contacting the authors and following a review process that ensures ethical considerations are met.

## Contributing

We welcome contributions to enhance the functionality and usability of this project. If you are interested in contributing, please follow these steps:

1. **Fork the Repository**: Click on the "Fork" button at the top right corner of this page to create a copy of the repository on your GitHub account.
2. **Clone the Forked Repository**: Use `git clone` to clone the repository to your local machine.
3. **Create a New Branch**: Create a new branch for your feature or bug fix.
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. **Make Changes**: Implement your feature or fix the bug.
5. **Commit Changes**: Commit your changes with a descriptive commit message.
   ```bash
   git commit -am "Add new feature: your feature name"
   ```
6. **Push to GitHub**: Push your changes to your forked repository.
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Submit a Pull Request**: Go to the original repository and create a pull request from your forked repository.

Please ensure that your code follows the project's coding conventions and includes relevant tests and documentation.

## License

This project is licensed under the [MIT License](LICENSE).

## References

Please refer to the original paper for detailed information on the methodology and findings:

- Park, J. S., Zou, C. Q., Shaw, A., Hill, B. M., Cai, C., Morris, M. R., Willer, R., Liang, P., & Bernstein, M. S. (2024). *Generative Agent Simulations of 1,000 People*.

## Acknowledgements
We thank Akaash Kolluri (Github: akaashkolluri) for the help setting up this open source repository. 

In addition, we thank Douglas Guilbeault, Amir Goldberg, Diyi Yang, Jeff Hancock, Serina Chang for their insights and discussions. 


## Contact

For questions or inquiries, please contact the corresponding author:

- **Joon Sung Park**: [joonspk@stanford.edu](mailto:joonspk@stanford.edu)
