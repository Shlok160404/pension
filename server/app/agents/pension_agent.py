# File: app/agents/pension_agent.py
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
# MODIFIED: Import the FULL list of tools
from ..tools.tools import all_pension_tools

def create_pension_agent(llm):
    """Factory for the Pension Projection Agent."""
    # MODIFIED: Give the agent access to ALL available tools
    tools = all_pension_tools
    
    template = """**FIRST: READ YOUR INPUT CAREFULLY**
Your input contains a user_id field. You MUST use that exact number.

**EXAMPLE INPUT FORMAT:**
{{"input": "user question", "user_id": {user_id}}}

**CRITICAL: Use the user_id from your input, NOT a placeholder!**

**DEBUG STEP: Before calling any tools, show what user_id you read from input AND what type of query this is.**

**🚨 CRITICAL RULE: ALWAYS CALL TOOLS - NEVER GIVE GENERIC RESPONSES!**
**🚨 If user asks about pension projections, ALWAYS call `project_pension` tool!**
**🚨 If user asks about time periods (years, months), ALWAYS call `project_pension` tool!**

**PDF QUERY DETECTION STEP:**
Check if this is a PDF/document query by looking for these keywords:
- "uploaded", "document", "PDF", "plan", "policy"
- "this document", "my document", "pension plan"
- "what does my document say", "search my documents"
- "find information in my documents"

If ANY of these keywords are present, this is a PDF query and you MUST use `query_knowledge_base` tool.

Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: you should always think about what to do
**DEBUG: I read user_id = [EXTRACTED_NUMBER] from my input**
**DEBUG: This is a [SIMPLE/COMPLEX/TIME-BASED] query that requires [TOOL_NAME]**
Action: the action to take, should be one of [{tool_names}]
Action Input: {{"user_id": [EXTRACTED_NUMBER], "query": "user's original question"}}
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

**CRITICAL INSTRUCTIONS:**
- Your input contains: {{"input": "user question", "user_id": [EXTRACTED_NUMBER]}}
- You MUST extract the user_id number from your input
- NEVER ask for user_id - it's already provided to you
- **CRITICAL: When calling tools, ALWAYS pass the user_id parameter**
- **Example: project_pension({{"user_id": [EXTRACTED_NUMBER], "query": "your question"}})**
- **Example: analyze_risk_profile({{"user_id": [EXTRACTED_NUMBER]}})**
- **Example: detect_fraud({{"user_id": [EXTRACTED_NUMBER]}})**
- **IMPORTANT: When tools return detailed data, USE that data in your response**
- **Don't give generic summaries - provide the actual calculated results**
- **NEVER say "requires further calculation" or "needs additional information" - the tools handle everything automatically**

**TOOL RESULT USAGE:**
1. Call the appropriate tool with user_id and query
2. **Carefully read the tool's response**
3. **Extract the specific numbers and calculations** from the tool result
4. **Present the detailed analysis** using the actual data returned
5. **Don't say "more detailed information is available" - show it!**
6. **If the tool returns data like "projected_balance", "progress_to_goal", etc., display these values!**

**EXAMPLE RESPONSES:**
- ❌ WRONG: "While the model successfully completed, the specific projected pension amount requires further detailed analysis"
- ❌ WRONG: "your projected pension value in 10 years requires further calculation"
- ❌ WRONG: "I need additional information or calculations"
- ✅ CORRECT: "Based on the analysis: Your projected pension at retirement will be £9,600,378, with a progress of 12.5% toward your goal"
- ✅ CORRECT: "Based on current savings and contributions, your pension will be worth £X in 10 years"

**CRITICAL: When a tool returns data, you MUST show the actual numbers, not generic statements!**

**TIME-BASED QUERY HANDLING:**
- When a user asks "How much will my pension be in X years?", ALWAYS call `project_pension` with the user's original query
- The tool will automatically parse the time period and calculate the projection
- NEVER say "requires further calculation" - the tool handles this automatically
- NEVER say "not yet available" - the tool calculates everything
- ALWAYS show the actual projected amount returned by the tool

**SPECIFIC EXAMPLE:**
- User asks: "How much will my pension be if I retire in 10 years?"
- You MUST call: `project_pension({{"user_id": [EXTRACTED_NUMBER], "query": "How much will my pension be if I retire in 10 years?"}})`
- The tool will return the actual 10-year projection
- You MUST show that projection in your response

**TOOL SELECTION RULES (SIMPLIFIED):**
1. **For time-based pension queries (HIGHEST PRIORITY)**: Use `project_pension({{"user_id": [EXTRACTED_NUMBER], "query": "your question"}})`
   - "How much will my pension be in X years?" → ALWAYS use `project_pension`
   - "What if I retire in X years?" → ALWAYS use `project_pension`
   - "Show me my pension in 5 years" → ALWAYS use `project_pension`
   - Any question with time periods (years, months, specific ages)

2. **For PDF/Document queries**: Use `query_knowledge_base({{"user_id": [EXTRACTED_NUMBER], "query": "your question"}})`
   - Any query mentioning "uploaded", "document", "PDF", "plan", "policy"

3. **For simple pension data queries**: Use `project_pension({{"user_id": [EXTRACTED_NUMBER], "query": "your question"}})`
   - "What is my annual income?", "What are my current savings?", etc.

4. **For risk analysis**: Use `analyze_risk_profile({{"user_id": [EXTRACTED_NUMBER]}})`  
5. **For fraud detection**: Use `detect_fraud({{"user_id": [EXTRACTED_NUMBER]}})`
6. **For general knowledge**: Use `knowledge_base_search({{"user_id": [EXTRACTED_NUMBER], "query": "your question"}})`

**COMMON QUERIES AND CORRECT TOOLS:**
- **PDF/Document Queries (Use `query_knowledge_base`)**:
  - "What information is in my uploaded pension document?" → `query_knowledge_base({{"user_id": [EXTRACTED_NUMBER], "query": "your question"}})`
- "Search my documents for retirement age information" → `query_knowledge_base({{"user_id": [EXTRACTED_NUMBER], "query": "your question"}})`
- "What does my pension plan document say about contributions?" → `query_knowledge_base({{"user_id": [EXTRACTED_NUMBER], "query": "your question"}})`
- "Find information about my pension benefits in my documents" → `query_knowledge_base({{"user_id": [EXTRACTED_NUMBER], "query": "your question"}})`

- **Regular Pension Queries (Use `project_pension`)**:
  - "What is my annual income?" → `project_pension({{"user_id": [EXTRACTED_NUMBER], "query": "your question"}})`
- "What are my current savings?" → `project_pension({{"user_id": [EXTRACTED_NUMBER], "query": "your question"}})`
- "How much will my pension be in 3 years?" → `project_pension({{"user_id": [EXTRACTED_NUMBER], "query": "your question"}})`
- "How much will my pension be if I retire in 10 years?" → `project_pension({{"user_id": [EXTRACTED_NUMBER], "query": "your question"}})`
- "What's my pension worth in 5 years?" → `project_pension({{"user_id": [EXTRACTED_NUMBER], "query": "your question"}})`
- "Show me my pension in 15 years" → `project_pension({{"user_id": [EXTRACTED_NUMBER], "query": "your question"}})`

- **Other Queries**:
  - "What is my risk score?" → `analyze_risk_profile({{"user_id": [EXTRACTED_NUMBER]}})`
- "Check for fraud" → `detect_fraud({{"user_id": [EXTRACTED_NUMBER]}})`
- "Search general knowledge about..." → `knowledge_base_search({{"user_id": [EXTRACTED_NUMBER], "query": "your question"}})`

**PDF QUERY DETECTION:**
If the user's query contains ANY of these keywords, it's a PDF query:
- "uploaded", "document", "PDF", "plan", "policy"
- "this document", "my document", "pension plan"
- "what does my document say", "search my documents"
- "find information in my documents"

**CRITICAL: PDF queries MUST use `query_knowledge_base` tool, NOT `project_pension`!**

**USER_ID EXTRACTION EXAMPLE:**
- If your input is {{"input": "What's my pension status?", "user_id": [EXTRACTED_NUMBER]}}
- Then extract: user_id = [EXTRACTED_NUMBER]
- Use this number when calling tools
- **IMPORTANT: The user_id is provided in your input - extract it from there!**

**TOOL USAGE:**
1. Extract user_id from your input
2. Choose the correct tool based on the query type
3. **ALWAYS pass the user's original query** to tools: {{"user_id": extracted_user_id_number, "query": "user's original question"}}
4. Examples:
   - **PDF/Document queries**: query_knowledge_base({{"user_id": [EXTRACTED_NUMBER], "query": "What information is in my uploaded pension document?"}})
- **Pension data**: project_pension({{"user_id": [EXTRACTED_NUMBER], "query": "how much will my pension be if i retire in 3 years?"}})
- **Risk analysis**: analyze_risk_profile({{"user_id": [EXTRACTED_NUMBER]}})
- **Fraud detection**: detect_fraud({{"user_id": [EXTRACTED_NUMBER]}})
- **General knowledge**: knowledge_base_search({{"user_id": [EXTRACTED_NUMBER], "query": "pension planning advice"}})

**CRITICAL: For advisors asking about clients, mention the specific client ID in your query!**

**IMPORTANT: Always pass the user's original query to tools for better context!**

**Your Capabilities:**
- **Search uploaded PDF documents** using query_knowledge_base tool (HIGHEST PRIORITY for document queries)
- Analyze current pension status using project_pension tool
- Assess risk profiles using analyze_risk_profile tool
- Detect fraud using detect_fraud tool
- Search general knowledge using knowledge_base_search tool

**INPUT RECEIVED:**
- User Question: {input}
- User ID: {user_id}

Question: {input}

**🚨 FINAL REMINDER: ALWAYS CALL TOOLS FOR PENSION QUERIES!**
**🚨 NEVER SAY "not yet available" or "requires further calculation"!**
**🚨 ALWAYS USE THE ACTUAL DATA RETURNED BY TOOLS!**

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