import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

def get_tool_definitions() -> list[dict]:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_python",
                "description": "Execute a Python code snippet in an isolated subprocess and return stdout and stderr. Use this to test code, run computations, and verify solutions. Has a 30-second timeout and 10,000 character output limit. Prefer this over reasoning about what code will do — always run it to confirm.",  # YOUR CODE HERE
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute"
                        }
                    },
                    "required": ["code"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a file from the local filesystem and return it as a string. Use this to inspect existing files before modifying or analyzing them. Returns an error message if the file does not exist.",  # YOUR CODE HERE
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {
                            "type": "string",
                            "description": "Path to the file to read"
                        }
                    },
                    "required": ["filepath"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file at the given path. Use this to save the final verified solution as a .py file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filepath": {"type": "string", "description": "Path to write the file to"},
                        "content": {"type": "string", "description": "Content to write"}
                    },
                    "required": ["filepath", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Signal that the task is complete and return the final answer or solution. Call this only when you have verified the solution is correct by running it. Do not call this until all tests pass.",  # YOUR CODE HERE
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The final answer or completed solution"
                        }
                    },
                    "required": ["answer"]
                }
            }
        },
    ]
    return tools

def run_python(code: str, timeout: int = 30, max_output_chars: int = 10000) -> str:
    """
    Execute Python code in a subprocess and return the output.

    Args:
        code: Python source code to execute
        timeout: Maximum execution time in seconds
        max_output_chars: Maximum characters to return (truncate beyond this)

    Returns:
        str: Combined stdout/stderr output, or error message.
             Format:
               On success: "STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
               On timeout: "ERROR: Execution timed out after {timeout} seconds"
               On error:   "ERROR: {error_message}"

    Implementation steps:
        1. Create a temporary .py file using tempfile.NamedTemporaryFile
           (use suffix='.py', mode='w', delete=False)
        2. Write the code string to the file, then close it
        3. Use subprocess.run() with:
           - ["python3", tmpfile_path] as the command
           - capture_output=True, text=True, timeout=timeout
        4. Catch subprocess.TimeoutExpired -> return timeout error message
        5. Catch any other Exception -> return generic error message
        6. On success: combine stdout and stderr into the output string
        7. If combined output exceeds max_output_chars, truncate and append
           "\n... [OUTPUT TRUNCATED]"
        8. Clean up the temp file in a finally block using os.unlink()
    """
    # YOUR CODE HERE
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.py',mode='w',delete=False) as f:
            f.write(code)
            tmp_path = f.name
        result = subprocess.run(
        ["python3", tmp_path],
            capture_output=True,   # captures stdout and stderr instead of printing them
            text=True,             # returns strings instead of bytes
            timeout=timeout        # raises subprocess.TimeoutExpired if it runs too long
        )
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        if len(output) > max_output_chars:
            output = output[:max_output_chars] + "\n... [OUTPUT TRUNCATED]"

        return output
    except subprocess.TimeoutExpired:
        return f"ERROR: Execution timed out after {timeout} seconds"
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        if tmp_path is not None:
            os.unlink(tmp_path)
    
def read_file(filepath: str, max_chars: int = 10000) -> str:
    """
    Read a file and return its contents.

    Args:
        filepath: Path to the file to read
        max_chars: Maximum characters to return

    Returns:
        str: File contents, or error message if file not found / unreadable.
             On success: the file text (truncated if needed)
             On not found: "ERROR: File not found: {filepath}"
             On other error: "ERROR: Could not read file: {error_message}"
    """
    # YOUR CODE HERE
    try:
        with open(filepath,mode='r') as f:
            content = f.read()
        if len(content) > max_chars:
            content = content[:max_chars]+ "\n... [OUTPUT TRUNCATED]"
        return content
    except FileNotFoundError:
        return f"ERROR: File not found: {filepath}"
    except Exception as e:
        return f"Could not read file: {e}"

def write_file(filepath: str, content: str) -> str:
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        return f"File written successfully: {filepath}"
    except Exception as e:
        return f"ERROR: Could not write file: {e}"

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """
    Route a tool call to the appropriate implementation.

    Args:
        tool_name: One of "run_python", "read_file", "finish"
        tool_input: Dict of arguments matching the tool input_schema

    Returns:
        str: The tool output string

    Raises:
        ValueError: If tool_name is not recognized

    Implementation:
        - If tool_name == "run_python": call run_python(tool_input["code"])
        - If tool_name == "read_file": call read_file(tool_input["filepath"])
        - If tool_name == "finish": return f"TASK COMPLETE: {tool_input['answer']}"
        - Otherwise: raise ValueError(f"Unknown tool: {tool_name}")
    """
    # YOUR CODE HERE
    if tool_name == 'run_python':
        return run_python(tool_input['code'])
    elif tool_name == 'read_file':
        return read_file(tool_input['filepath'])
    elif tool_name == 'finish':
        return f"TASK COMPLETE: {tool_input['answer']}"
    elif tool_name == "write_file":
        return write_file(tool_input["filepath"], tool_input["content"])
    else:
        raise ValueError(f"Unknown tool: {tool_name}")


@dataclass
class AgentStep:
    """Record of a single thought-action-observation cycle."""
    step_number: int
    thought: Optional[str]          # The model reasoning (from text blocks)
    tool_name: Optional[str]        # Which tool was called
    tool_input: Optional[dict]      # Arguments passed to the tool
    observation: Optional[str]      # Result from tool execution
    raw_response: Optional[dict] = None  # Full API response for debugging


@dataclass
class AgentState:
    """Complete state of an agent run."""
    task: str                                   # The original task description
    messages: list = field(default_factory=list) # Anthropic API message history
    steps: list = field(default_factory=list)    # List of AgentStep records
    is_done: bool = False                        # Whether the agent has finished
    final_answer: Optional[str] = None           # The answer from the finish tool
    total_input_tokens: int = 0                  # Running token count
    total_output_tokens: int = 0

def get_system_prompt() -> str:
    """
    Returns the system prompt that instructs the LLM how to behave as a
    ReAct coding agent.

    The prompt should:
    - Define the agent role (coding assistant that solves problems step by step)
    - Instruct it to think before acting (ReAct pattern)
    - Describe when to use each tool (run_python, read_file, finish)
    - Tell it to test code by running it, not just writing it
    - Instruct it to handle errors by examining the error and retrying
    - Tell it to call finish() with the final answer when done

    Returns:
        str: The system prompt
    """
    # YOUR CODE HERE
    prompt = """You are a coding agent that solves programming problems step by step.

            For every task, think carefully before acting. Reason about what you need to do, then use the appropriate tool.

            Tools:
            - run_python: Execute Python code to test solutions and verify they work. Always run code to confirm it works — do not just reason about what it will do.
            - read_file: Read an existing file before modifying or analyzing it.
            - write_file: Save content to a file at the given path. Only use this if the task explicitly mentions saving or writing a file. If you do write a file, ensure it contains only the final polished solution — no test scaffolding or temporary code. Call this before finish.
            - finish: Call this when the task is complete with the verified final answer. Do not call finish until all tests pass.

            Error handling:
            - If a tool returns an error or failure, read the error message carefully and retry with a different approach.
            - If code raises an exception, debug it by examining the traceback and fixing the issue.

            Keep your reasoning concise. Do not write long explanations between actions.
            """
    return prompt

def truncate_context(
    messages: list[dict],
    keep_first: int = 1,
    keep_last_steps: int = 4
) -> list[dict]:
    """
    Truncate message history to fit in context window.

    Args:
        messages: Full message history (list of API message dicts)
        keep_first: Number of initial messages to always keep (the task)
        keep_last_steps: Number of recent agent steps to keep.
                         Each step = 2 messages (assistant + tool_result),
                         so this keeps keep_last_steps * 2 messages from the end.

    Returns:
        list[dict]: Truncated message history with a summary marker in the middle.

    If total messages <= keep_first + keep_last_steps * 2:
        return messages unchanged (no truncation needed).

    Otherwise, return:
        messages[:keep_first]
        + [{"role": "user", "content": "[Earlier steps omitted for brevity. "
            "Focus on the current state and the original task.]"}]
        + messages[-(keep_last_steps * 2):]
    """
    # YOUR CODE HERE
    if len(messages) <= keep_first + keep_last_steps*2:
        return messages
    else:
        return(
            messages[:keep_first]
            + [{"role": "user", "content": "[Earlier steps omitted for brevity. "
                "Focus on the current state and the original task.]"}]
            + messages[-(keep_last_steps * 2):]
        )
    
def detect_loop(steps: list[AgentStep], window: int = 3) -> bool:
    """
    Detect if the agent is stuck in a loop.

    Args:
        steps: List of completed AgentStep records
        window: Number of recent steps to check for repetition

    Returns:
        bool: True if the last `window` steps all used the same tool with
              the same input (indicating a loop), False otherwise.

    Edge cases:
        - If len(steps) < window, return False (not enough data)
        - Compare both tool_name and tool_input for equality
    """
    # YOUR CODE HERE
    if len(steps) < window:
        return False
    else:
        stuck = True
        for i in range(window-1):
            if steps[-i-1].tool_input != steps[-i-2].tool_input or steps[-i-1].tool_name != steps[-i-2].tool_name:
                stuck = False
        return stuck
