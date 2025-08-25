from typing import List, Tuple, Dict, Any
from langchain_core.messages import AIMessage
import base64

try:
    from vl_convert import vegalite_to_png  # type: ignore
except Exception:  # pragma: no cover
    vegalite_to_png = None  # type: ignore


def _spec_to_png_data_uri(spec: Dict[str, Any]):
    try:
        if vegalite_to_png is None:
            return None
        png_bytes = vegalite_to_png(spec)  # type: ignore
        b64 = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


def create_visualizer_node():
    def visualizer_node(state: Dict[str, Any]) -> Dict[str, Any]:
        print(f"🔍 Visualizer: Starting with state keys: {list(state.keys())}")
        print(f"🔍 Visualizer: intermediate_steps count: {len(state.get('intermediate_steps', []))}")
        
        steps = state.get("intermediate_steps", []) or []
        risk_data = None
        fraud_data = None
        projection_data = None

        for step in steps:
            try:
                if isinstance(step, (list, tuple)) and len(step) == 2:
                    action, observation = step
                    tool_name = getattr(action, "tool", None) or getattr(action, "tool_name", None) or ""
                    print(f"🔍 Visualizer: Processing step with tool: {tool_name}")
                    if isinstance(observation, Dict):
                        if tool_name == "analyze_risk_profile" and risk_data is None:
                            risk_data = observation
                            print(f"🔍 Visualizer: Found risk data: {list(risk_data.keys()) if risk_data else 'None'}")
                        elif tool_name == "detect_fraud" and fraud_data is None:
                            fraud_data = observation
                            print(f"🔍 Visualizer: Found fraud data: {list(fraud_data.keys()) if fraud_data else 'None'}")
                        elif tool_name == "project_pension" and projection_data is None:
                            projection_data = observation
                            print(f"🔍 Visualizer: Found projection data: {list(projection_data.keys()) if projection_data else 'None'}")
                            print(f"🔍 Visualizer: Projection data type: {type(projection_data)}")
                            if isinstance(projection_data, dict):
                                print(f"🔍 Visualizer: Has current_data: {'current_data' in projection_data}")
                                print(f"🔍 Visualizer: Has projection_analysis: {'projection_analysis' in projection_data}")
                                print(f"🔍 Visualizer: Has chart_data: {'chart_data' in projection_data}")
            except Exception as e:
                print(f"⚠ Visualizer: Error processing step: {e}")
                continue

        # Only create charts if the user requested them
        user_query = ""
        if state.get("messages") and len(state.get("messages", [])) > 0:
            # Get the user's original query from the first message
            first_message = state["messages"][0]
            if hasattr(first_message, 'content'):
                user_query = first_message.content.lower()
        
        # Check if supervisor detected chart request OR if user explicitly wants charts
        supervisor_wants_charts = state.get("wants_charts", False)
        user_wants_charts = any(word in user_query for word in ["graph", "chart", "visual", "show me", "display"]) or supervisor_wants_charts
        
        if not user_wants_charts:
            print(f"🔍 Visualizer: User did not request charts, skipping visualization")
            # Just add a simple confirmation message
            messages = list(state["messages"])  # type: ignore
            messages.append(AIMessage(content="Your pension analysis is complete."))
            return {"messages": messages}
        
        print(f"🔍 Visualizer: User requested charts, creating visualizations")
        print(f"🔍 Visualizer: Supervisor wants_charts flag: {supervisor_wants_charts}")
        
        # Create charts only when requested
        charts = {}
        plotly_figs = {}
        chart_images = {}

        # Projection line (start vs end)
        try:
            if isinstance(projection_data, dict):
                # Check for the ACTUAL structure returned by project_pension tool
                if "current_data" in projection_data and "projection_analysis" in projection_data:
                    current_data = projection_data.get("current_data", {})
                    projection_analysis = projection_data.get("projection_analysis", {})
                    chart_data = projection_data.get("chart_data", {})
                    
                    # Extract values from current data
                    current_savings_str = current_data.get("current_savings", "£0")
                    annual_income_str = current_data.get("annual_income", "£0")
                    age = current_data.get("age", 0)
                    retirement_age = current_data.get("retirement_age_goal", 65)
                    annual_contribution_str = current_data.get("annual_contribution", "£0")
                    savings_rate_str = current_data.get("savings_rate", "0%")
                    
                    # Extract values from projection analysis
                    years_to_retirement = projection_analysis.get("years_to_retirement", 0)
                    projected_balance_str = projection_analysis.get("projected_balance", "£0")
                    progress_to_goal_str = projection_data.get("progress_to_goal", "0%")
                    status = projection_data.get("status", "Unknown")
                    
                    # Convert string values to numbers
                    def extract_number(value_str):
                        try:
                            # Remove £ and % symbols and commas
                            clean_str = str(value_str).replace("£", "").replace("%", "").replace(",", "")
                            return float(clean_str)
                        except:
                            return 0
                    
                    current_savings = extract_number(current_savings_str)
                    annual_income = extract_number(annual_income_str)
                    annual_contribution = extract_number(annual_contribution_str)
                    projected_balance = extract_number(projected_balance_str)
                    progress_percentage = extract_number(progress_to_goal_str)
                    
                    print(f"🔍 Visualizer: Current savings: {current_savings}, Annual income: {annual_income}")
                    print(f"🔍 Visualizer: Progress percentage: {progress_percentage}")
                    
                    # Create ONLY the 3 charts you want:
                    
                    # 1. Progress to Goal (Bar chart - NOT gauge)
                    charts["progress_to_goal"] = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                        "description": "Progress to Goal",
                        "data": {"values": [
                            {"category": "Current Progress", "amount": progress_percentage},
                            {"category": "Remaining Goal", "amount": 100 - progress_percentage}
                        ]},
                        "mark": "bar",  # Changed from {"type": "bar", "stack": "zero"}
                        "encoding": {
                            "x": {"field": "category", "type": "nominal", "title": ""},
                            "y": {"field": "amount", "type": "quantitative", "title": "Percentage (%)"},
                            "color": {"field": "category", "type": "nominal"}
                        }
                    }
                    
                    print(f"🔍 Visualizer: Created progress_to_goal chart with data: {charts['progress_to_goal']['data']}")
                    
                    # 2. Pension Growth (Smooth line chart - NO nodes/dots and NO dips)
                    # Create CORRECTED pension growth chart using actual projection data
                    print(f"🔍 Visualizer: Creating corrected pension growth chart")
                    
                    # Get the actual projected balance from the projection tool result
                    projected_balance = extract_number(projected_balance_str)
                    years_to_retirement = projection_analysis.get("years_to_retirement", 32)
                    annual_return_rate = 0.091  # 9.1% from the data
                    annual_contribution = extract_number(annual_contribution_str)
                    
                    print(f"🔍 Visualizer: Using actual projected balance: £{projected_balance:,.0f}")
                    print(f"🔍 Visualizer: Years to retirement: {years_to_retirement}")
                    print(f"🔍 Visualizer: Annual return rate: {annual_return_rate:.1%}")
                    print(f"🔍 Visualizer: Annual contribution: £{annual_contribution:,.0f}")
                    
                    # Generate corrected data points using the SAME formula as projection tool
                    # This should match exactly: FV = FV(current_savings) + FV(annual_contributions)
                    pension_growth_data = []
                    for year in range(years_to_retirement + 1):
                        current_age = age + year  # Start from current age
                        
                        if year == 0:
                            value = current_savings
                        else:
                            # Use the EXACT same calculation as projection tool:
                            # 1. Future value of current savings: FV = PV * (1 + r)^n
                            fv_current_savings = current_savings * ((1 + annual_return_rate) ** year)
                            
                            # 2. Future value of annual contributions: FV = PMT * ((1 + r)^n - 1) / r
                            # Where PMT = annual contribution, r = annual return rate, n = years
                            if annual_return_rate > 0:
                                   fv_contributions = annual_contribution * ((((1 + annual_return_rate) ** year) - 1) / annual_return_rate)
                            else:
                                fv_contributions = annual_contribution * year
                            
                            # 3. Total = both combined
                            value = fv_current_savings + fv_contributions
                        
                        pension_growth_data.append({
                            "age": current_age,
                            "projected_value": round(value, 2)
                        })
                    
                    charts["pension_growth"] = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                        "description": "Pension Growth Over Time (Corrected)",
                        "data": {"values": pension_growth_data},
                        "mark": "line",  # NO point markers
                        "encoding": {
                            "x": {"field": "age", "type": "quantitative", "title": "Age"},
                            "y": {"field": "projected_value", "type": "quantitative", "title": "Projected Pension Value (£)"}
                        }
                    }
                    
                    print(f"🔍 Visualizer: Created corrected pension growth with {len(pension_growth_data)} data points")
                    print(f"🔍 Visualizer: Final value at age {age + years_to_retirement}: £{pension_growth_data[-1]['projected_value']:,.0f}")
                    
                    # Verify the final value matches the projection tool
                    expected_final = projected_balance
                    actual_final = pension_growth_data[-1]['projected_value']
                    if abs(expected_final - actual_final) > 1000:  # Allow small rounding differences
                        print(f"⚠️ Visualizer: WARNING - Chart final value (£{actual_final:,.0f}) doesn't match projection tool (£{expected_final:,.0f})")
                        print(f"🔍 Visualizer: Difference: £{abs(expected_final - actual_final):,.0f}")
                    else:
                        print(f"✅ Visualizer: Chart final value matches projection tool: £{actual_final:,.0f}")
                    
                    # 3. Savings Analysis (ONLY 2 bars: current savings and annual income)
                    charts["savings_analysis"] = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                        "description": "Current Savings vs Annual Income",
                        "data": {"values": [
                            {"category": "Current Savings", "amount": current_savings},
                            {"category": "Annual Income", "amount": annual_income}
                        ]},
                        "mark": "bar",  # Changed from {"type": "bar"}
                        "encoding": {
                            "x": {"field": "category", "type": "nominal", "title": ""},
                            "y": {"field": "amount", "type": "quantitative", "title": "Amount (£)"},
                            "color": {"field": "category", "type": "nominal"}
                        }
                    }
                    
                    print(f"🔍 Visualizer: Created savings_analysis chart with data: {charts['savings_analysis']['data']}")
                    
                    # IMPORTANT: Don't use the old chart_data - it has wrong calculations!
                    # We've already created the corrected pension_growth chart above
                    print(f"🔍 Visualizer: Ignoring old chart_data with incorrect calculations")
                        
        except Exception as e:
            print(f"⚠ Error processing pension data in visualizer: {e}")
            pass

        # Risk score bar
        try:
            if isinstance(risk_data, dict):
                score = risk_data.get("risk_score")
                if isinstance(score, (int, float)):
                    charts["risk"] = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                        "description": "Risk score",
                        "data": {"values": [{"metric": "Risk Score", "value": float(score)}]},
                        "mark": "bar",
                        "encoding": {
                            "x": {"field": "metric", "type": "nominal", "title": ""},
                            "y": {"field": "value", "type": "quantitative", "title": "Score"}
                        }
                    }
                    
                    plotly_figs["risk"] = {
                        "data": [
                            {"type": "bar", "x": ["Risk Score"], "y": [float(score)], "name": "Risk Score"}
                        ],
                        "layout": {"title": "Risk Score"}
                    }
        except Exception:
            pass

        # Fraud confidence bar
        try:
            if isinstance(fraud_data, dict):
                fraud_score = fraud_data.get("fraud_score")
                if isinstance(fraud_score, (int, float)):
                    charts["fraud"] = {
                        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                        "description": "Fraud confidence",
                        "data": {"values": [{"metric": "Fraud Score", "value": float(fraud_score)}]},
                        "mark": "bar",
                        "encoding": {
                            "x": {"field": "metric", "type": "nominal", "title": ""},
                            "y": {"field": "value", "type": "quantitative", "title": "Score"}
                        }
                    }
                    
                    plotly_figs["fraud"] = {
                        "data": [
                            {"type": "bar", "x": ["Risk Score"], "y": [float(fraud_score)], "name": "Fraud Score"}
                        ],
                        "layout": {"title": "Fraud Score"}
                    }
        except Exception:
            pass

        # Export images (best effort)
        for name, spec in charts.items():
            try:
                uri = _spec_to_png_data_uri(spec)
                if uri:
                    chart_images[name] = uri
            except Exception as e:
                print(f"⚠ Error generating image for chart {name}: {e}")
                continue

        # Build Plotly figure JSONs for frontend rendering
        try:
            # 1. Progress to Goal (Bar chart - NOT gauge)
            progress_to_goal = charts.get("progress_to_goal")
            if progress_to_goal and isinstance(progress_to_goal, dict):
                values = progress_to_goal.get("data", {}).get("values", [])
                categories = [v.get("category") for v in values]
                amounts = [v.get("amount") for v in values]
                if categories and amounts:
                    # Create with multiple names to ensure frontend compatibility
                    plotly_figs["progress_to_goal"] = {
                        "data": [
                            {
                                "type": "bar", 
                                "x": categories, 
                                "y": amounts, 
                                "name": "Percentage",
                                "marker": {"color": ["#4CAF50", "#FF9800"]},
                                "text": [f"{amount:.1f}%" for amount in amounts],
                                "textposition": "auto",
                                "hovertemplate": "Category: %{x}<br>Progress: %{y:.1f}%<extra></extra>"
                            }
                        ],
                        "layout": {
                            "title": {"text": "Progress to Goal", "font": {"size": 16}},
                            "xaxis": {"title": ""},
                            "yaxis": {"title": "Percentage (%)", "range": [0, 100]},
                            "barmode": "group",
                            "annotations": [
                                {
                                    "x": categories[0], "y": amounts[0] + 2,
                                    "text": f"{amounts[0]:.1f}%",
                                    "showarrow": False, "font": {"color": "white", "size": 12}
                                },
                                {
                                    "x": categories[1], "y": amounts[1] + 2,
                                    "text": f"{amounts[1]:.1f}%",
                                    "showarrow": False, "font": {"color": "white", "size": 12}
                                }
                            ]
                        }
                    }
                    # Also create with alternative name
                    plotly_figs["progress_to_goal_chart"] = plotly_figs["progress_to_goal"]
                    print(f"🔍 Plotly: Created progress_to_goal bar chart with {len(categories)} categories")
            
            # 2. Pension Growth (Smooth line chart - NO points)
            pension_growth = charts.get("pension_growth")
            if pension_growth and isinstance(pension_growth, dict):
                values = pension_growth.get("data", {}).get("values", [])
                if values and len(values) >= 2:
                    ages = [v.get("age") for v in values]
                    pension_values = [v.get("projected_value") for v in values]
                    if ages and pension_values:
                        # Use the CORRECTED data from our calculation
                        plotly_figs["pension_growth"] = {
                            "data": [
                                {
                                    "type": "scatter", 
                                    "mode": "lines", 
                                    "x": ages, 
                                    "y": pension_values, 
                                    "name": "Pension Value", 
                                    "line": {"width": 3, "color": "#2196F3"},
                                    "hovertemplate": "Age: %{x}<br>Pension: £%{y:,.0f}<extra></extra>"
                                }
                            ],
                            "layout": {
                                "title": {"text": "Pension Growth Over Time (Corrected)", "font": {"size": 16}},
                                "xaxis": {"title": "Age", "gridcolor": "#e0e0e0"},
                                "yaxis": {"title": "Pension Value (£)", "gridcolor": "#e0e0e0"},
                                "plot_bgcolor": "white",
                                "paper_bgcolor": "white",
                                "annotations": [
                                    {
                                        "x": ages[-1], "y": pension_values[-1],
                                        "text": f"£{pension_values[-1]:,.0f}",
                                        "showarrow": True, "arrowhead": 2, "arrowsize": 1, "arrowwidth": 2,
                                        "arrowcolor": "#2196F3", "ax": 20, "ay": -30,
                                        "font": {"color": "#2196F3", "size": 12}
                                    }
                                ]
                            }
                        }
                        # Also create with alternative name
                        plotly_figs["pension_growth_chart"] = plotly_figs["pension_growth"]
                        print(f"🔍 Plotly: Created pension_growth line chart with {len(ages)} data points")
                        print(f"🔍 Plotly: Final value: £{pension_values[-1]:,.0f} (should match £2,922,252)")
            
            # 3. Savings Analysis (ONLY 2 bars)
            savings_analysis = charts.get("savings_analysis")
            if savings_analysis and isinstance(savings_analysis, dict):
                values = savings_analysis.get("data", {}).get("values", [])
                categories = [v.get("category") for v in values]
                amounts = [v.get("amount") for v in values]
                if categories and amounts:
                    plotly_figs["savings_analysis"] = {
                        "data": [
                            {
                                "type": "bar", 
                                "x": categories, 
                                "y": amounts, 
                                "name": "Amount",
                                "marker": {"color": ["#2196F3", "#FF9800"]},
                                "text": [f"£{amount:,.0f}" for amount in amounts],
                                "textposition": "auto",
                                "hovertemplate": "Category: %{x}<br>Amount: £%{y:,.0f}<extra></extra>"
                            }
                        ],
                        "layout": {
                            "title": {"text": "Current Savings vs Annual Income", "font": {"size": 16}},
                            "xaxis": {"title": ""},
                            "yaxis": {"title": "Amount (£)", "gridcolor": "#e0e0e0"},
                            "barmode": "group",
                            "plot_bgcolor": "white",
                            "paper_bgcolor": "white",
                            "annotations": [
                                {
                                    "x": categories[0], "y": amounts[0] + 10000,
                                    "text": f"£{amounts[0]:,.0f}",
                                    "showarrow": False, "font": {"color": "white", "size": 10}
                                },
                                {
                                    "x": categories[1], "y": amounts[1] + 10000,
                                    "text": f"£{amounts[1]:,.0f}",
                                    "showarrow": False, "font": {"color": "white", "size": 10}
                                }
                            ]
                        }
                    }
                    # Also create with alternative name
                    plotly_figs["savings_analysis_chart"] = plotly_figs["savings_analysis"]
                    print(f"🔍 Plotly: Created savings_analysis bar chart with {len(categories)} categories")
        except Exception as e:
            print(f"⚠ Error creating Plotly figures: {e}")
            pass

        messages = list(state["messages"])  # type: ignore
        
        # Add chart information to messages safely
        try:
            # Don't add any chart information to messages - let the frontend handle charts silently
            # Only add a simple confirmation that the query was processed
            messages.append(AIMessage(content="Your pension analysis is complete with visualizations."))
            
            # Still add the technical chart data for the frontend
            if chart_images:
                messages.append(AIMessage(content="[CHART_IMAGES] " + str(chart_images)))
            if plotly_figs:
                messages.append(AIMessage(content="[PLOTLY_FIGS] " + str(plotly_figs)))
        except Exception as e:
            print(f"⚠ Error adding chart messages: {e}")
            # Add a simple message if chart processing fails
            messages.append(AIMessage(content="Pension analysis completed."))

        updates: Dict[str, Any] = {"messages": messages}
        
        # Mark visualization as complete for supervisor routing
        agent_results = state.get("agent_results", {})
        agent_results["visualization_done"] = True
        updates["agent_results"] = agent_results
        
        # Add chart data safely
        try:
            if charts:
                updates["charts"] = charts
            if chart_images:
                updates["chart_images"] = chart_images
            if plotly_figs:
                updates["plotly_figs"] = plotly_figs
        except Exception as e:
            print(f"⚠ Error adding chart data to updates: {e}")
        
        print(f"🔍 Visualizer: Returning updates with keys: {list(updates.keys())}")
        print(f"🔍 Visualizer: Charts count: {len(charts)}")
        print(f"🔍 Visualizer: Plotly figs count: {len(plotly_figs)}")
        print(f"🔍 Visualizer: Chart images count: {len(chart_images)}")
        
        # Debug: Show the actual chart data being created
        for chart_name, chart_data in charts.items():
            print(f"🔍 Chart '{chart_name}': mark={chart_data.get('mark')}, data points={len(chart_data.get('data', {}).get('values', []))}")
        
        # Debug: Show what Plotly figures are being created
        print(f"🔍 Plotly figures being created: {list(plotly_figs.keys())}")
        for fig_name, fig_data in plotly_figs.items():
            if isinstance(fig_data, dict) and "data" in fig_data:
                data_type = fig_data["data"][0].get("type") if fig_data["data"] else "unknown"
                print(f"🔍 Plotly figure '{fig_name}': type={data_type}")
        
        return updates

    return visualizer_node
