# demo_multi_agent_load_balancer.py
"""
================================================================================
            Multi-Agent Load Balancer Demo (vLLM Backend)
================================================================================

This demo showcases multiple fair_llm agents running in PARALLEL, all hitting
the same vllm_load_manager backend simultaneously.

It demonstrates:
- Multiple agents sharing a distributed LLM infrastructure
- Concurrent execution via asyncio
- Load balancing across vLLM GPU nodes

ARCHITECTURE:
=============

    +-------------+  +-------------+  +-------------+  +-------------+
    |  Agent #1   |  |  Agent #2   |  |  Agent #3   |  |  Agent #4   |
    +------+------+  +------+------+  +------+------+  +------+------+
           |                |                |                |
           +----------------+----------------+----------------+
                                    |
                                    v
                       +------------+------------+
                       |   LoadBalancerAdapter   |
                       +------------+------------+
                                    |
                                    v
                       +------------+------------+
                       |   vllm_load_manager     |
                       |      (Port 8123)        |
                       +------------+------------+
                                    |
                    +---------------+---------------+
                    v               v               v
              [GPU Node 1]    [GPU Node 2]    [GPU Node N]

PREREQUISITES:
==============
1. vllm_load_manager running
2. At least one vllm_load_node registered
3. Model available in supported models list
"""

import asyncio
import time

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

# =============================================================================
# CONFIGURATION - Modify these as needed
# =============================================================================
MANAGER_IP = "localhost"
MANAGER_PORT = 8123
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
NUM_AGENTS = 4
# =============================================================================


def create_calculator_agent(llm, agent_id: int) -> SimpleAgent:
    """Create a calculator agent with a unique ID."""
    tool_registry = ToolRegistry()
    tool_registry.register_tool(SafeCalculatorTool())

    executor = ToolExecutor(tool_registry)
    memory = WorkingMemory()

    planner = SimpleReActPlanner(llm, tool_registry)
    planner.prompt_builder.role_definition = RoleDefinition(
        f"You are Calculator Agent #{agent_id}. Perform mathematical calculations. "
        "Use the safe_calculator tool to compute results."
    )

    return SimpleAgent(
        llm=llm,
        planner=planner,
        tool_executor=executor,
        memory=memory,
        max_steps=5
    )


async def run_agent(agent: SimpleAgent, agent_id: int, task: str) -> dict:
    """Run a single agent and return results with timing."""
    print(f"[Agent {agent_id}] Starting: {task}")
    start = time.time()

    try:
        response = await agent.arun(task)
        elapsed = time.time() - start
        print(f"[Agent {agent_id}] Done in {elapsed:.2f}s")
        return {"agent_id": agent_id, "task": task, "response": response, "time": elapsed, "success": True}
    except Exception as e:
        elapsed = time.time() - start
        print(f"[Agent {agent_id}] Failed: {e}")
        return {"agent_id": agent_id, "task": task, "response": str(e), "time": elapsed, "success": False}


async def main():
    print("=" * 70)
    print("       Multi-Agent Load Balancer Demo")
    print("=" * 70)
    print(f"Manager: http://{MANAGER_IP}:{MANAGER_PORT}")
    print(f"Model: {MODEL}")
    print(f"Agents: {NUM_AGENTS}")
    print("=" * 70)

    # Initialize the load balancer adapter
    print("\nConnecting to vllm_load_manager...")
    llm = LoadBalancerAdapter(
        manager_ip=MANAGER_IP,
        port=MANAGER_PORT,
        model=MODEL,
        timeout=900,
        verbose=True,
        preload_model=True
    )

    # Show cluster status
    print("\nCluster Status:")
    for node, info in llm.get_health_status().items():
        print(f"  {node}: {info.get('status')} | Model: {info.get('model')}")

    # Preset tasks - each agent gets a different math problem
    tasks = [
        "What is 25 * 17?",
        "Calculate 144 / 12 + 50",
        "What is 99 + 101?",
        "Compute 15 * 15 - 25",
    ]

    # Create agents
    print(f"\nCreating {NUM_AGENTS} agents...")
    agents = [create_calculator_agent(llm, i + 1) for i in range(NUM_AGENTS)]

    # Run ALL agents in parallel
    print("\n" + "=" * 70)
    print("Running all agents IN PARALLEL...")
    print("=" * 70 + "\n")

    start_time = time.time()

    results = await asyncio.gather(*[
        run_agent(agent, i + 1, tasks[i % len(tasks)])
        for i, agent in enumerate(agents)
    ])

    total_time = time.time() - start_time

    # Results summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for r in results:
        status = "OK" if r["success"] else "FAIL"
        print(f"\nAgent {r['agent_id']} [{status}] ({r['time']:.2f}s)")
        print(f"  Task: {r['task']}")
        print(f"  Answer: {str(r['response'])[:100]}")

    # Stats
    successful = sum(1 for r in results if r["success"])
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Agents: {NUM_AGENTS}")
    print(f"  Successful: {successful}/{NUM_AGENTS}")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Avg per Agent: {total_time / NUM_AGENTS:.2f}s (parallel)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
