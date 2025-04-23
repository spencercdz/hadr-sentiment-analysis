"""
ReportBuilderAgent - Responsible for building the final report from summarized data
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from .base import ReportWorkflowState, ASSETS_DIR, TEMPLATES_DIR

# Import report builder tool
import sys
from pathlib import Path
TOOLS_DIR = Path(__file__).parent / "tools"
sys.path.append(str(TOOLS_DIR))
import tools.build_report as report_builder

class ReportBuilderAgent:
    """Agent responsible for building the final report."""
    def __init__(self, llm, workflow_state: ReportWorkflowState, assets_path: Optional[Path] = None):
        self.llm = llm
        self.workflow_state = workflow_state
        self.assets_path = assets_path or ASSETS_DIR
        self.json_structure_template = workflow_state.json_structure_template
        
        # Ensure the output directory exists
        (self.assets_path / "outputs").mkdir(exist_ok=True)
    
    def build_report(self):
        """Build the final report using the summarized data."""
        try:
            # Check if summarized data already has the right structure
            required_keys = ["sections", "tweets", "details"]
            missing_keys = [key for key in required_keys if key not in self.workflow_state.summarized_data]
            
            # If the structure is already correct, use it directly
            if not missing_keys:
                print("Summarized data already has the correct structure, using it directly")
                self.workflow_state.report_data = self.workflow_state.summarized_data
            else:
                # If not, use the LLM to generate proper report structure
                print(f"Summarized data missing keys {missing_keys}, generating complete report")
                # Create a properly formatted prompt for the model
                prompt = (
                    f"Generate a complete report based on the summarized data.\n"
                    f"Your output MUST be a valid JSON object with EXACTLY these three top-level keys: \"sections\", \"tweets\", and \"details\".\n\n"
                    f"The output format must follow this template EXACTLY:\n"
                    f"{self.json_structure_template}\n\n"
                    f"Based on this input data:\n"
                    f"{json.dumps(self.workflow_state.summarized_data, indent=4)}\n\n"
                    f"Your response should be ONLY valid JSON with no additional text.\n"
                    f"JSON output:"
                )
                
                result = self.llm(prompt)
                
                try:
                    # Clean the result and extract JSON
                    import re
                    clean_result = re.sub(r'<think>[\s\S]*?</think>', '', result)
                    json_match = re.search(r'\{[\s\S]*\}', clean_result)
                    
                    if json_match:
                        clean_result = json_match.group(0)
                        parsed_data = json.loads(clean_result)
                        self.workflow_state.report_data = parsed_data
                    else:
                        # Use the summarized data directly as a fallback
                        print("Could not extract JSON from LLM output, using summarized data directly")
                        self.workflow_state.report_data = self.workflow_state.summarized_data
                except Exception as json_error:
                    print(f"Error parsing LLM output: {str(json_error)}")
                    # Use the summarized data directly as a fallback
                    self.workflow_state.report_data = self.workflow_state.summarized_data
            
            # Save the report data to a file in the correct format for report generation
            outputs_dir = self.assets_path / "outputs"
            outputs_dir.mkdir(exist_ok=True)  # Ensure outputs directory exists
            
            # Create a more concise and readable filename
            # Extract disaster type and location for a more descriptive filename
            disaster_type = self.workflow_state.disaster_type or "disaster"
            disaster_loc = self.workflow_state.disaster_location or "location"
            date_str = self.workflow_state.disaster_year or datetime.now().strftime("%Y")
            
            base_filename = f"hadr_{disaster_type}_{disaster_loc}_{date_str}"
            json_filename = f"{base_filename}.json"
            pdf_filename = f"{base_filename}.pdf"
            
            # Save in standardized locations
            report_path = outputs_dir / json_filename
            output_pdf_path = outputs_dir / pdf_filename
            
            # Save JSON data
            with open(report_path, 'w') as f:
                json.dump(self.workflow_state.report_data, f, indent=4)
            
            # Store the output paths in the workflow state
            self.workflow_state.output_path = report_path
            self.workflow_state.report_path = output_pdf_path
            
            # Create a standardized "latest" copy for easy reference
            latest_json_path = outputs_dir / "latest_report_data.json"
            try:
                with open(report_path, 'r') as src, open(latest_json_path, 'w') as dst:
                    dst.write(src.read())
            except Exception as e:
                print(f"Note: Could not create latest copy: {e}")
                
            # Generate the PDF report using the improved build_report.py that directly reads from JSON
            try:
                print("Using the improved build_report.py that reads directly from the JSON file")
                
                # Import the custom version directly to use its structure
                import importlib.util
                import sys
                
                # Get the module spec
                spec = importlib.util.spec_from_file_location(
                    "build_report", 
                    str(TOOLS_DIR / "build_report.py")
                )
                
                # Create the module
                build_report_module = importlib.util.module_from_spec(spec)
                
                # Execute the module
                spec.loader.exec_module(build_report_module)
                
                # Use the generate_report function that reads directly from the JSON file
                # It will access the sections, tweets, and details from the JSON structure
                output_pdf_path = build_report_module.generate_report(
                    json_file_path=str(report_path),
                    output_pdf_path=str(output_pdf_path)
                )
                
                print(f"Report generated successfully at: {output_pdf_path}")
                
                # Return success with both JSON and PDF paths
                return {
                    "success": True, 
                    "message": "Successfully generated report using JSON structure", 
                    "report_path": str(output_pdf_path),
                    "json_path": str(report_path)
                }
            except Exception as pdf_error:
                print(f"Error generating PDF: {str(pdf_error)}")
                return {"success": False, "message": f"Failed to generate PDF report: {str(pdf_error)}"}
        except Exception as e:
            import traceback
            print(f"Error in build_report: {str(e)}")
            print(traceback.format_exc())
            return {"success": False, "message": str(e)}
