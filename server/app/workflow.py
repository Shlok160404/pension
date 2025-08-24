# File: app/workflow.py
from typing import TypedDict, List, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from functools import partial
from langgraph.graph.message import add_messages

# Import all our modular components
from .tools.tools import all_pension_tools
from .agents.risk_agent import create_risk_agent
from .agents.fraud_agent import create_fraud_agent
from .agents.pension_agent import create_pension_agent
from .agents.summarizer_agent import create_summarizer_chain
from .agents.visualizer_agent import create_visualizer_node
from .agents.supervisor import create_supervisor_chain


# --- State Definition ---
class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str
    intermediate_steps: List[BaseMessage]
    turns: int
    charts: dict
    chart_images: dict
    plotly_figs: dict  # Changed from plotly_figures to match visualizer agent
    final_response: dict
    user_id: int  # Add user_id to state


# --- Graph Builder Function ---
def build_agent_workflow():
    """
    Builds the LangGraph workflow by creating instances of all agents
    and wiring them together.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # Create all agent and supervisor runnables
    risk_agent_runnable = create_risk_agent(llm)
    fraud_agent_runnable = create_fraud_agent(llm)
    projection_agent_runnable = create_pension_agent(llm)
    summarizer_chain_runnable = create_summarizer_chain(llm)
    supervisor_chain_runnable = create_supervisor_chain(llm)

    # --- Supervisor Node ---
    def supervisor_node(state: AgentState):
        # Simple routing based on query content
        user_query = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                user_query = msg.content
                break
            elif isinstance(msg, dict) and str(msg.get("role", "")) == "user":
                user_query = msg.get("content", "")
                break
            elif isinstance(msg, tuple) and len(msg) >= 2 and str(msg[0]) == "user":
                user_query = str(msg[1])
                break
        
        query_lower = user_query.lower()
        
        # Check if user wants charts/visualizations
        wants_charts = any(word in query_lower for word in ["graph", "chart", "visual", "show me", "display", "plot"])
        
        # Simple routing logic
        if any(word in query_lower for word in ["risk", "volatility", "diversity", "debt"]):
            next_value = "risk_analyst"
        elif any(word in query_lower for word in ["fraud", "suspicious", "anomaly", "transaction"]):
            next_value = "fraud_detector"
        elif any(word in query_lower for word in ["projection", "growth", "future", "years", "retire", "savings", "income", "contribution"]):
            next_value = "projection_specialist"
        else:
            next_value = "projection_specialist"  # Default to pension specialist
        
        print(f"🔍 Supervisor: Routing to {next_value}")
        print(f"🔍 Supervisor: User wants charts: {wants_charts}")
        
        # Store chart request flag in state for later use
        return {"next": next_value, "wants_charts": wants_charts}

    # --- Generic Agent Runner ---
    def agent_node(state: AgentState, agent_runnable):
        # Find the latest user message content
        last_user_text = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                last_user_text = msg.content
                break
            if isinstance(msg, tuple) and len(msg) >= 2 and str(msg[0]).lower() in ("user", "human"):
                last_user_text = msg[1]
                break
            if isinstance(msg, dict) and str(msg.get("role", "").lower()) in ("user", "human"):
                last_user_text = msg.get("content")
                break

        if not last_user_text:
            return {"messages": list(state["messages"]) + [AIMessage(content="⚠️ No user message found to process.")]}

        # Get user_id from workflow state and pass it to the agent
        user_id = state.get("user_id")
        
        # Create input with user_id for the agent
        agent_input = {
            "input": last_user_text
        }
        
        # If user_id is available, add it to the input so agents can access it
        if user_id:
            agent_input["user_id"] = user_id
            print(f"🔍 Workflow: Passing user_id={user_id} to agent")
        else:
            # Try to get user_id from context if not in state
            try:
                from .tools.tools import get_current_user_id_from_context
                context_user_id = get_current_user_id_from_context()
                if context_user_id:
                    user_id = context_user_id
                    agent_input["user_id"] = user_id
                    print(f"🔍 Workflow: Got user_id={user_id} from context")
                else:
                    print(f"⚠️ Workflow: No user_id found in state or context")
            except Exception as e:
                print(f"⚠️ Workflow: Error getting user_id from context: {e}")

        result = agent_runnable(agent_input)

        # Normalize result into messages and capture intermediate steps
        new_messages: List[BaseMessage] = list(state["messages"])
        new_intermediate_steps = list(state.get("intermediate_steps", []))
        
        final_text = None
        if isinstance(result, str):
            final_text = result
        elif isinstance(result, dict):
            final_text = result.get("output") or result.get("content") or result.get("text")
            # Capture intermediate steps from agent result
            if result.get("intermediate_steps"):
                new_intermediate_steps.extend(result["intermediate_steps"])
        else:
            final_text = str(result)

        if final_text:
            new_messages.append(AIMessage(content=final_text))

        # Return both messages and intermediate steps
        updates = {"messages": new_messages}
        if new_intermediate_steps:
            updates["intermediate_steps"] = new_intermediate_steps
        return updates

    # --- Summarizer Node ---
    def summarizer_node(state: AgentState):
        # Call the summarizer function directly
        summary_result = summarizer_chain_runnable(state)
        
        # Extract the final response if available
        final_response = summary_result.get("final_response", {})
        
        if final_response:
            # Add the structured final response
            new_messages = list(state["messages"])
            new_messages.append(AIMessage(content=final_response.get("summary", "Summary completed.")))
            # Store the final response in state for frontend access
            return {
                "messages": new_messages,
                "final_response": final_response
            }
        else:
            # Fallback to old behavior
            if isinstance(summary_result, str):
                summary_text = summary_result
            elif isinstance(summary_result, dict):
                summary_text = summary_result.get("output") or summary_result.get("content") or summary_result.get("text") or str(summary_result)
            else:
                summary_text = getattr(summary_result, "content", None) or str(summary_result)
            
            return {"messages": state["messages"] + [AIMessage(content=summary_text)]}

    # --- Visualization Node ---
    from .agents.visualizer_agent import create_visualizer_node as _make_vis
    visualizer_node = _make_vis()

    # --- Build the graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)

    # Specific nodes from generic agent_node
    workflow.add_node("risk_analyst", partial(agent_node, agent_runnable=risk_agent_runnable))
    workflow.add_node("fraud_detector", partial(agent_node, agent_runnable=fraud_agent_runnable))
    workflow.add_node("projection_specialist", partial(agent_node, agent_runnable=projection_agent_runnable))

    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("visualizer", visualizer_node)

    # --- Wire up the graph ---
    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges("supervisor", lambda x: x["next"], {
        "risk_analyst": "risk_analyst",
        "fraud_detector": "fraud_detector",
        "projection_specialist": "projection_specialist",
        "summarizer": "summarizer",
        "visualizer": "visualizer",
    })

    # After specialist agents -> go to visualizer (for charts)
    workflow.add_edge("risk_analyst", "visualizer")
    workflow.add_edge("fraud_detector", "visualizer")
    workflow.add_edge("projection_specialist", "visualizer")

    # After visualizer -> go to summarizer
    workflow.add_edge("visualizer", "summarizer")

    # After summarizer -> workflow ends
    workflow.add_edge("summarizer", END)

    return workflow.compile()


# Compile and make available
graph = build_agent_workflow()
print("✅ Simple multi-agent graph compiled successfully.")


def save_graph_image():
    """Generates and saves a PNG image of the compiled graph."""
    try:
        graph_viz = graph.get_graph()
        try:
            image_data = graph_viz.draw_mermaid_png()
        except AttributeError:
            image_data = graph_viz.draw_png()
        with open("pension_agent_supervisor_graph.png", "wb") as f:
            f.write(image_data)
        print("\n✅ Graph visualization saved to 'pension_agent_supervisor_graph.png'")
    except ImportError as e:
        print(f"\n❌ ERROR: Could not generate graph image. Please install prerequisites.")
        print("   System-level: 'graphviz' (e.g., 'sudo apt-get install graphviz')")
        print("   Python packages: 'pip install pygraphviz Pillow'")
        print(f"   Original error: {e}")
    except Exception as e:
        print(f"\n❌ An error occurred while generating the graph: {e}")


if __name__ == "__main__":
    save_graph_image()