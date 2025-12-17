# demo_single_agent_calculator_load_balancer.py
import asyncio

"""
This script demonstrates a single autonomous agent using the load balancer backend.

It is a variant of demo_single_agent_calculator.py that uses LoadBalancerAdapter to connect
to a distributed vLLM cluster for inference instead of running a local HuggingFace model.
"""

# --- Step 1: Import the necessary framework components ---
from fairlib import (
    LoadBalancerAdapter,
    ToolRegistry,
    SafeCalculatorTool,
    ToolExecutor,
    WorkingMemory,
    SimpleAgent, 
    SimpleReActPlanner,
    RoleDefinition
)

async def main():
    """
    The main function to assemble and run our single agent.
    """
    print("=== Single Agent Calculator (Load Balancer Backend) ===\n")
    
    # --- Step 1: Get connection details from user ---
    manager_ip = input("Enter load manager IP address: ").strip()
    port_str = input("Enter load manager port: ").strip()
    port = int(port_str)
    
    print("\nInitializing a single agent with load balancer backend...")

    # --- Step 2: Assemble the Agent's "Anatomy" ---

    # a) The "Brain": The Language Model via Load Balancer
    llm = LoadBalancerAdapter(
        manager_ip=manager_ip,
        model="mistralai/Mistral-7B-Instruct-v0.3",
        port=port,
        max_tokens=1024,
        temperature=0.7
    )

    # b) The "Toolbelt": The Tool Registry and Tools
    tool_registry = ToolRegistry()
    
    calculator_tool = SafeCalculatorTool()
    tool_registry.register_tool(calculator_tool)
    
    print(f"Agent's tools: {[tool.name for tool in tool_registry.get_all_tools().values()]}")

    # c) The "Hands": The Tool Executor
    executor = ToolExecutor(tool_registry)

    # d) The "Memory": The Agent's Short-Term Memory
    memory = WorkingMemory()

    # e) The "Mind": The Planner
    planner = SimpleReActPlanner(llm, tool_registry)

    # Modify the default role:
    planner.prompt_builder.role_definition = \
    RoleDefinition(
        "You are an expert mathematical calculator. Your job it is to perform mathematical calculations.\n"
        "You reason step-by-step to determine the best course of action. If a user's request requires "
        "multiple steps or tools, you must break it down and execute them sequentially. You must follow the strict formatting rules that follow..."
    )

    # --- Step 3: Create the Agent ---
    agent = SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=memory,
        max_steps=10
    )
    print("Agent successfully created. You can now chat with the agent.")
    print("Try asking it a math problem, like 'What is 45 * 11?' or 'What is the result of 1024 divided by 256?'. Type 'exit' to quit.")

    # --- Step 4: Run the Interaction Loop ---
    while True:
        try:
            user_input = input("\n👤 You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("🤖 Agent: Goodbye!")
                break
            
            agent_response = await agent.arun(user_input)
            print(f"LLM Raw Output:\n{agent_response}")
            print(f"🤖 Agent: {agent_response}")

        except KeyboardInterrupt:
            print("\n🤖 Agent: Exiting...")
            break


if __name__ == "__main__":
    asyncio.run(main())

