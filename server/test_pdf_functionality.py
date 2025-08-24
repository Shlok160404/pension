#!/usr/bin/env python3
"""
Test PDF ingestion functionality with user's actual PDF
"""

import sys
import os

def test_pdf_ingestion(pdf_path=None):
    """Test the complete PDF ingestion pipeline"""
    try:
        print("🧪 Testing PDF Ingestion Pipeline")
        print("=" * 50)
        
        # Import required modules
        from app.file_ingestion import ingest_pdf_to_chroma
        from app.chromadb_service import get_or_create_collection, query_collection
        from app.tools.tools import set_request_user_id, clear_request_user_id
        
        # Simulate user context
        user_id = 9
        set_request_user_id(user_id)
        print(f"✅ Set user context for user {user_id}")
        
        # Get PDF path from user or auto-detect
        if pdf_path:
            test_file = pdf_path
            print(f"📄 Using provided PDF path: {test_file}")
        else:
            # Auto-detect PDF files
            pdf_files = [f for f in os.listdir('.') if f.lower().endswith('.pdf')]
            
            if not pdf_files:
                print("❌ No PDF files found in current directory!")
                print("📁 Please provide a PDF path or place your PDF file in this directory.")
                print("💡 Usage: python test_pdf_ingestion.py [PDF_PATH]")
                return
            
            print(f"📄 Found PDF files: {pdf_files}")
            test_file = pdf_files[0]
            print(f"📄 Using auto-detected PDF: {test_file}")
        
        # Verify file exists
        if not os.path.exists(test_file):
            print(f"❌ File not found: {test_file}")
            print("📁 Please check the file path and try again.")
            return
        
        print(f"✅ File found: {test_file}")
        print(f"📊 File size: {os.path.getsize(test_file)} bytes")
        
        # Step 1: Ingest the PDF
        print(f"\n📊 Step 1: Ingesting PDF '{test_file}' into ChromaDB...")
        result = ingest_pdf_to_chroma(test_file, user_id)
        print(f"✅ Ingestion result: {result}")
        
        if result.get("status") != "success":
            print("❌ PDF ingestion failed!")
            return
        
        # Step 2: Test knowledge base search
        print("\n🔍 Step 2: Testing knowledge base search...")
        collection = get_or_create_collection(f"user_{user_id}_docs")
        
        # Test different queries (you can modify these based on your PDF content)
        test_queries = [
            "pension",
            "balance",
            "contribution",
            "investment",
            "risk",
            "fee",
            "policy"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: '{query}'")
            results = query_collection(collection, query, n_results=2)
            
            if results and 'documents' in results and results['documents']:
                print(f"📚 Found {len(results['documents'])} relevant chunks:")
                for i, doc in enumerate(results['documents']):
                    print(f"   {i+1}. {doc[:100]}...")
            else:
                print("❌ No relevant chunks found")
        
        # Step 3: Test the knowledge_base_search tool
        print("\n🔧 Step 3: Testing knowledge_base_search tool...")
        from app.tools.tools import knowledge_base_search
        
        # Test with a general query about your document
        tool_result = knowledge_base_search("What information is available in my uploaded document?")
        print(f"📋 Tool result: {tool_result}")
        
        # Step 4: Test with workflow
        print("\n🚀 Step 4: Testing with AI workflow...")
        from app.workflow import graph
        
        # Test query about uploaded document
        test_query = "Based on my uploaded pension document, What information is available in my uploaded document??"
        print(f"🤖 AI Query: '{test_query}'")
        
        # Pass user_id in the workflow state
        result = graph.invoke({
            'messages': [('user', test_query)],
            'user_id': user_id  # Include user_id in the state
        })
        print(f"✅ Workflow completed!")
        print(f"📊 Result keys: {list(result.keys())}")
        
        # Check for final response
        if 'final_response' in result:
            final_response = result['final_response']
            print(f"📝 Summary: {final_response.get('summary', 'No summary')}")
            
            # Check for charts
            charts = final_response.get('charts', {})
            plotly_figs = final_response.get('plotly_figs', {})
            
            if charts or plotly_figs:
                print(f"📊 Charts generated: {list(charts.keys())}")
                print(f"📊 Plotly figures: {list(plotly_figs.keys())}")
        
        clear_request_user_id()
        print("\n✅ PDF ingestion test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check for command line argument
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"🎯 Using PDF path from command line: {pdf_path}")
        test_pdf_ingestion(pdf_path)
    else:
        # Auto-detect PDF in current directory
        print("🔍 No PDF path provided, auto-detecting PDF files...")
        test_pdf_ingestion()
