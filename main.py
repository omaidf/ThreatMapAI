import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import shutil
import logging

# Import utility modules
from utils.common import success_msg, error_msg, warning_msg, info_msg
from utils.file_utils import check_output_directory
from utils.model_utils import validate_model_path

# Configure logging only if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from repository_analyzer.analyzer import RepositoryAnalyzer
from repository_analyzer.embedding_store import EmbeddingStore
from llm_processor.processor import LLMProcessor
from visualizer.visualizer import ThreatModelVisualizer
from visualizer.code_diagram import CodeDiagramGenerator

app = FastAPI(title="AI Threat Model Map Generator")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RepositoryRequest(BaseModel):
    repo_url: str
    local_path: Optional[str] = None

@app.post("/analyze")
async def analyze_repository(request: RepositoryRequest):
    try:
        # Initialize components
        repo_path = request.local_path or "temp_repo"
        
        # Ensure output directory exists
        output_dir = os.getenv("OUTPUT_DIR", "output")
        check_output_directory(output_dir)
        
        embedding_store = EmbeddingStore()
        analyzer = RepositoryAnalyzer(repo_path)
        llm_processor = LLMProcessor(embedding_store)
        visualizer = ThreatModelVisualizer()
        diagram_generator = CodeDiagramGenerator()

        # Clone and analyze repository
        analyzer.clone_repository(request.repo_url)
        analysis_results = analyzer.analyze_code()

        # Generate threat model
        threat_model = llm_processor.generate_threat_model(analysis_results)

        # Initialize tracking for visualization outputs
        class_diagram_path = None
        flow_diagram_path = None
        report_path = None
        visualization_errors = []

        # Generate visualizations
        logger.info("Generating visualizations...")
        try:
            class_diagram = diagram_generator.generate_class_diagram(analysis_results["components"])
            flow_diagram = diagram_generator.generate_flow_diagram(analysis_results["data_flows"])
            
            # Save diagrams
            class_diagram_path = diagram_generator.save_diagram(class_diagram, "class_diagram.mmd")
            flow_diagram_path = diagram_generator.save_diagram(flow_diagram, "flow_diagram.mmd")
        except Exception as e:
            error_msg(f"Failed to generate diagrams: {str(e)}")
            visualization_errors.append(f"Failed to generate diagrams: {str(e)}")

        # Generate report
        logger.info("Generating report...")
        try:
            report_path = visualizer.generate_report(threat_model)
        except Exception as e:
            error_msg(f"Failed to generate report: {str(e)}")
            visualization_errors.append(f"Failed to generate report: {str(e)}")

        # Clean up temporary repository if it was created
        if not request.local_path and os.path.exists(repo_path):
            shutil.rmtree(repo_path)

        # Include any visualization errors in the response
        response = {
            "status": "success",
            "report_path": report_path,
            "class_diagram_path": class_diagram_path,
            "flow_diagram_path": flow_diagram_path,
            "threat_model": threat_model
        }
        
        if visualization_errors:
            response["visualization_errors"] = visualization_errors
            response["status"] = "partial_success"

        return response

    except Exception as e:
        # Clean up on error
        if not request.local_path and os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        error_msg(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    # Ensure model directory exists
    os.makedirs("models", exist_ok=True)
    
    info_msg("Starting AI Threat Model Map Generator API server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 