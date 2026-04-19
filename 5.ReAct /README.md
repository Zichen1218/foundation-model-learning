# Project: ReAct Coding Agent

**New concept:** The agent reasoning loop  

---

## What this project builds

A ReAct (Reasoning + Acting) agent that solves coding problems by interleaving chain-of-thought reasoning with tool use in a structured **thought → action → observation** loop. The agent is powered by DeepSeek-V3 via API and can execute Python code, read files, write files, and signal task completion.

The core insight of ReAct: letting the model reason out loud before acting improves both action quality and interpretability. Pure chain-of-thought can reason but cannot verify. Pure tool use is brittle without planning. ReAct combines both — reason about what to do, do it, observe the result, reason about what happened.

---

## File structure

```
05_react_coding_agent/
├── 05_react_coding_agent.ipynb   # Main notebook with teaching cells and sanity checks
├── functions.py                  # All tool definitions, implementations, and agent utilities
├── run_agent.py                  # Entry point — runs the production agent from the terminal
└── README.md
```

---

## How it works

### The agent loop

```
while not done:
    thought = llm.generate("Think about what to do next")
    action  = llm.generate("Choose a tool and arguments")
    observation = execute(action)        # run the tool in your environment
    context.append(thought, action, observation)
```

This is structurally identical to GPT's autoregressive loop — but with external interrupts. After each generation, control returns to your code, which executes the action and injects the result back before the next generation step.

### Tools available to the agent

| Tool | What it does |
|---|---|
| `run_python` | Executes Python in an isolated subprocess, captures stdout/stderr, enforces 30s timeout |
| `read_file` | Reads a file from disk, returns contents or a clear error message |
| `write_file` | Writes content to a file — used only when the task explicitly requests it |
| `finish` | Signals task completion and returns the final verified answer |

### Robustness features (in `run_production_agent`)

- **Context truncation** — keeps the first message (original task) and the last N steps, dropping the middle when the history grows too long
- **Loop detection** — detects if the agent repeats the same tool call with the same arguments N times in a row and injects a nudge to try a different approach
- **Exponential backoff** — retries failed API calls with `2^attempt` second delays

---

## Setup

**API key:** Get a DeepSeek key at [platform.deepseek.com](https://platform.deepseek.com) and add it to a `.env` file in the project root:

```
DEEPSEEK_API_KEY=sk-...
```

**Install dependencies:**

```bash
pip install openai python-dotenv
```

---

## Usage

**From the terminal:**

```bash
python run_agent.py
```

You will be prompted to enter a task. Examples:

```
# Computation
What is the sum of the first 100 prime numbers? Use Python to compute this.

# Multi-step debugging
Write a Python function called flatten_nested that takes an arbitrarily nested
list and returns a flat list. Test it on at least 3 inputs.

# File reading + code generation
Read the file /tmp/students.csv and write a Python script that finds the
student with the highest score and calculates the average age.

# File output
Write a binary search implementation and save it to binary_search.py.
```

**From the notebook:**

Run cells sequentially. Each section has sanity checks that verify your implementation before moving to the next component.

---

## Key concepts for ML engineer interviews

**The agent loop is a program, not a model.** The LLM is a component inside your code. You control when it runs, what it sees, and what it can do.

**Prompt engineering is systems engineering.** The system prompt and tool descriptions are the agent's behavioral specification. Bad descriptions lead to wrong tool choices regardless of model quality — tool descriptions consume context tokens on every API call, so production systems curate them carefully.

**Context management is a first-class problem.** Every production agent system (Claude Code, Cursor, Devin) manages its context window explicitly. The truncation strategy here — keep the original task and the most recent steps — mirrors what real systems do.

**Tool design is API design.** The tools you give an agent define its capabilities. Clear, well-scoped tools with precise descriptions are the single biggest lever for agent reliability.

| Component | Production equivalent |
|---|---|
| `get_tool_definitions()` | Tool schemas in Claude API, OpenAI function calling |
| `dispatch_tool()` | MCP (Model Context Protocol) server dispatch |
| `run_python()` | Sandboxed execution (E2B, Modal, Docker) |
| `truncate_context()` | Context window management in Claude Code, Cursor |
| `detect_loop()` | Guardrails and circuit breakers in agent frameworks |
| `get_system_prompt()` | System prompts in every production LLM application |
| `run_production_agent()` | The agent loop in LangChain, CrewAI, Autogen |

---
