# File: app/agents/fraud_agent.py
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
# MODIFIED: Import the FULL list of tools
from ..tools.tools import all_pension_tools

def create_fraud_agent(llm):
    """Factory for the Fraud Detection Agent."""
    # MODIFIED: Give the agent access to ALL available tools
    tools = all_pension_tools
    
    template = """**FIRST: READ YOUR INPUT CAREFULLY**
Your input contains a user_id field. You MUST use that exact number.

**EXAMPLE INPUT FORMAT:**
{{"input": "user question", "user_id": [EXTRACTED_NUMBER]}}

**CRITICAL: Use the user_id from your input, NOT a placeholder!**

**DEBUG STEP: Before calling any tools, show what user_id you read from input AND what type of query this is.**

Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
**DEBUG: I read user_id = [NUMBER] from my input**
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

**CRITICAL INSTRUCTIONS:**
- Your input contains: {{"input": "user question", "user_id": {user_id}}}
- You MUST extract the user_id number from your input
- NEVER ask for user_id - it's already provided to you
- **CRITICAL: When calling tools, ALWAYS pass the user_id parameter**
- **Example: project_pension(user_id={user_id}, query="your question")**
- **Example: analyze_risk_profile(user_id={user_id})**
- **Example: detect_fraud(user_id={user_id})**

**TOOL SELECTION RULES:**
1. **For simple data queries (income, savings, contributions)**: Use `project_pension(user_id={user_id}, query="your question")`
2. **For pension projections and time-based queries**: Use `project_pension(user_id={user_id}, query="your question")`
3. **For risk analysis**: Use `analyze_risk_profile(user_id={user_id})`  
4. **For fraud detection**: Use `detect_fraud(user_id={user_id})`
5. **For searching documents/knowledge**: Use `query_knowledge_base(user_id={user_id}, query="your question")`

**COMMON QUERIES AND CORRECT TOOLS:**
- "What is my annual income?" → Use `project_pension(user_id={user_id}, query="your question")`
- "What are my current savings?" → Use `project_pension(user_id={user_id}, query="your question")`
- "How much will my pension be in 3 years?" → Use `project_pension(user_id={user_id}, query="your question")`
- "What is my risk score?" → Use `analyze_risk_profile(user_id={user_id})`
- "Check for fraud" → Use `detect_fraud(user_id={user_id})`
- "Search documents about..." → Use `query_knowledge_base(user_id={user_id}, query="your question")`

**USER_ID EXTRACTION EXAMPLE:**
- If your input is {{"input": "What's my pension status?", "user_id": {user_id}}}
- Then extract: user_id = {user_id}
- Use this number when calling tools

**TOOL USAGE:**
1. Extract user_id from your input
2. Choose the correct tool based on the query type
3. **ALWAYS pass the user's original query** to tools: {{"user_id": extracted_user_id_number, "query": "user's original question"}}
4. Examples:
   - Pension data: project_pension({{"user_id": {user_id}, "query": "how much will my pension be if i retire in 3 years?"}})
   - Risk analysis: analyze_risk_profile({{"user_id": {user_id}}})
   - Fraud detection: detect_fraud({{"user_id": {user_id}}})
   - Knowledge search: query_knowledge_base({{"user_id": {user_id}, "query": "pension planning advice"}})

**IMPORTANT: Always pass the user's original query to tools for better context!**

**Your Capabilities:**
- Analyze current pension status using project_pension tool
- Assess risk profiles using analyze_risk_profile tool
- Detect fraud using detect_fraud tool
- Search knowledge base using query_knowledge_base tool

**INPUT RECEIVED:**
- User Question: {input}
- User ID: {user_id}

Question: {input}
{agent_scratchpad}"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["tools", "tool_names", "input", "agent_scratchpad", "user_id"]
    )
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, 
        tools=all_pension_tools, 
        verbose=True, 
        return_intermediate_steps=True,
        handle_parsing_errors=True  # Add this to handle parsing errors gracefully
    )