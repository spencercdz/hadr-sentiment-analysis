"""
HADR Sentiment Analysis System - Main Entry Point

This serves as the main entry point for the HADR Sentiment Analysis system,
utilizing the LangGraph-based workflow for processing queries.
"""
import argparse
import sys
from pathlib import Path
from workflows import build_workflow

def main():
    """Main function to run the HADR agent system."""
    # Display startup banner
    print("=" * 80)
    print("HADR Sentiment Analysis System".center(80))
    print("Powered by LangGraph and DeepSeek-r1:8b".center(80))
    print("=" * 80)
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="HADR Agent System")
    parser.add_argument("--query", type=str, help="Query to process")
    args = parser.parse_args()
    
    # Build and compile the workflow
    print("Building workflow graph...")
    workflow = build_workflow()
    app = workflow.compile()
    
    if args.query:
        # Process the query from command line
        print(f"\nProcessing query: {args.query}")
        result = app.invoke({"query": args.query})
        print("\nResponse:")
        print(result["response"])
    else:
        # Interactive mode
        print("\nHADR Agent System - Interactive Mode")
        print("Type 'exit' to quit")
        
        while True:
            query = input("\nEnter your query: ")
            
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting...")
                break
            
            print(f"\nProcessing query: {query}")
            result = app.invoke({"query": query})
            print("\nResponse:")
            print(result["response"])
            
            # If a report was generated, notify the user
            if "report_path" in result and result["report_path"] and result["report_path"] != "Report generation failed or report not found.":
                try:
                    report_rel_path = Path(result["report_path"]).relative_to(Path.cwd())
                    print(f"\nReport available at: {report_rel_path}")
                except:
                    print(f"\nReport available at: {result['report_path']}")

if __name__ == "__main__":
    main()
