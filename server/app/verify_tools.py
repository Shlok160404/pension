#!/usr/bin/env python3
"""
Simple test script for the pension AI system
"""

def test_simple():
    """Simple test to check if the system works"""
    try:
        from app.workflow import graph
        from app.tools.tools import set_request_user_id, clear_request_user_id
        
        print("🧪 Simple Test - Pension AI System")
        print("=" * 40)
        
        # Test 1: Basic query with context
        print("📝 Test 1: Basic pension query")
        set_request_user_id(12)
        
        result = graph.invoke({'messages': [('user', "What's is a defined contributed plan?")]})
        print(f"✅ Workflow completed!")
        print(f"📊 Result keys: {list(result.keys())}")
        
        # Check for charts
        charts = result.get('charts', {})
        plotly_figs = result.get('plotly_figs', {})
        
        if charts or plotly_figs:
            print(f"📊 Charts: {list(charts.keys())}")
            print(f"📊 Plotly: {list(plotly_figs.keys())}")
            
            # Show actual JSON output
            if plotly_figs:
                print(f"\n🎨 JSON OUTPUT FOR FRONTEND:")
                for name, data in plotly_figs.items():
                    print(f"📈 {name}: {data}")
        else:
            print("📊 No charts generated")
        
        clear_request_user_id()
        print("\n" + "-" * 40)
        
        # Test 2: Chart request
        print("📝 Test 2: Chart request")
        set_request_user_id(18)
        
        result = graph.invoke({'messages': [('user', "analyse fraud?")]})
        print(f"✅ Workflow completed!")
        print(f"📊 Result keys: {list(result.keys())}")
        
        # Check for charts
        charts = result.get('charts', {})
        plotly_figs = result.get('plotly_figs', {})
        
        if charts or plotly_figs:
            print(f"📊 Charts: {list(charts.keys())}")
            print(f"📊 Plotly: {list(plotly_figs.keys())}")
            
            # Show actual JSON output
            if plotly_figs:
                print(f"\n🎨 JSON OUTPUT FOR FRONTEND:")
                for name, data in plotly_figs.items():
                    print(f"📈 {name}: {data}")
        else:
            print("📊 No charts generated")
        
        clear_request_user_id()
        print("\n" + "-" * 40)
        
        print("✅ Testing completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple()