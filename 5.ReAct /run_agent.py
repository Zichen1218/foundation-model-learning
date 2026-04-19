import os
import json
import time
from functions import get_tool_definitions,get_system_prompt,truncate_context,detect_loop,dispatch_tool,AgentState,AgentStep

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv('.env')  # loads .env from the current directory

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

MODEL = "deepseek-chat"


def run_production_agent(
    task: str,
    max_steps: int = 15,
    truncation_threshold: int = 10,
    keep_last_steps: int = 4,
    loop_window: int = 3,
    max_retries: int = 2
) -> AgentState:
    """
    Production-grade ReAct agent with all robustness features.

    Args:
        task: The coding task to solve
        max_steps: Maximum number of agent steps
        truncation_threshold: When to start truncating context
        keep_last_steps: Recent steps to keep during truncation
        loop_window: Steps to examine for loop detection
        max_retries: API call retries on failure

    Returns:
        AgentState with full history and metadata

    Implementation plan:
        1. Initialize state and tools (same as run_agent)
        2. Main loop (max_steps iterations):
           a. Check for loops -> if detected, inject nudge message
           b. Prepare messages (truncate if needed, but keep full history)
           c. Call API with retry logic:
              - Try up to max_retries + 1 times
              - On failure, wait 2^attempt seconds (exponential backoff)
              - If all retries fail, log error and break
           d. Parse response (same as run_agent)
           e. Execute tool and record step (same as run_agent)
           f. Check for finish -> break if done
        3. Print summary statistics
        4. Return state
    """
    # YOUR CODE HERE
    state = AgentState(task=task)
    state.messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": task}
    ]
    tools = get_tool_definitions()

    for step_num in range(1, max_steps + 1):

        # a. loop detection — inject a nudge if stuck
        if detect_loop(state.steps, window=loop_window):
            print(f"  ⚠️  Loop detected — injecting nudge")
            state.messages.append({
                "role": "user",
                "content": "You seem to be repeating the same action. Try a different approach."
            })

        # b. context truncation — truncated copy for API, full history in state
        if len(state.messages) > truncation_threshold:
            messages_for_api = truncate_context(state.messages, keep_last_steps=keep_last_steps)
        else:
            messages_for_api = state.messages

        # c. API call with exponential backoff retry
        response = None
        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    max_tokens=4096,
                    messages=messages_for_api,
                    tools=tools
                )
                break
            except Exception as e:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"  API error (attempt {attempt+1}/{max_retries+1}): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  API error: all retries failed. Stopping.")
                    return state

        state.total_input_tokens += response.usage.prompt_tokens
        state.total_output_tokens += response.usage.completion_tokens

        # d. parse response
        message = response.choices[0].message
        thought = message.content

        msg_dict = {"role": "assistant", "content": thought}
        if message.tool_calls:
            msg_dict["tool_calls"] = [tc.model_dump() for tc in message.tool_calls]
        state.messages.append(msg_dict)

        # e. execute tool and record step
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_input = json.loads(tool_call.function.arguments)
            tool_use_id = tool_call.id

            observation = dispatch_tool(tool_name, tool_input)

            state.messages.append({
                "role": "tool",
                "tool_call_id": tool_use_id,
                "content": observation
            })

            step = AgentStep(step_num, thought, tool_name, tool_input, observation)
            state.steps.append(step)

            print(f"Step {step_num}: {tool_name}")
            if thought:
                print(f"  Thought: {thought[:100]}")
            print(f"  Observation: {observation[:100]}")

            # f. check for finish
            if tool_name == "finish":
                state.is_done = True
                state.final_answer = tool_input.get("answer", "")
                break
        else:
            print(f"Step {step_num}: text-only response")
            if thought:
                print(f"  {thought[:200]}")

    # 3. summary
    print(f"\n{'='*50}")
    print(f"Task completed: {state.is_done}")
    print(f"Steps taken:    {len(state.steps)}")
    print(f"Tokens used:    {state.total_input_tokens} in, {state.total_output_tokens} out")
    if state.final_answer:
        print(f"Final answer:   {state.final_answer[:200]}")

    return state

task = input('write your task:\n')
state = run_production_agent(task)
print(f"\n{'='*50}")
print(f"Completed: {state.is_done}")
print(f"Steps: {len(state.steps)}")
print(f"Tokens: {state.total_input_tokens} in, {state.total_output_tokens} out")
if state.final_answer:
    print(f"\nFinal answer preview:\n{state.final_answer[:500]}...")