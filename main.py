from fastapi import FastAPI, UploadFile, Form, HTTPException, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from PIL import Image, ImageSequence
import torch
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor
import tempfile
from functional import seq
from typing import Tuple
import base64
from io import BytesIO
import os
import shutil

# Import ProcessingConfig, KosmosProcessor, and process_multipage_tiff from the appropriate module
from app import ProcessingConfig, KosmosProcessor, process_multipage_tiff  # Adjust the import path as necessary

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Mount static files with explicit directory check
if not (static_dir / "index.html").exists():
    raise RuntimeError(
        "Static files not found. Ensure the 'static' directory contains index.html"
    )

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Initialize model and processor
config = ProcessingConfig()
processor = KosmosProcessor(config)

@app.get("/")
async def read_index():
    try:
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Index file not found"
            )
        return FileResponse(
            str(index_path),
            media_type="text/html"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error serving index file: {str(e)}"
        )

@app.post("/api/process/")
async def process_image(
    file: UploadFile = File(...),
    task: str = Form(...),
    num_beams: int = Form(...),
    max_new_tokens: int = Form(...),
    temperature: float = Form(...)
):
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.tiff', '.tif')):
            raise HTTPException(
                status_code=400,
                detail="Only TIFF files are supported"
            )

        # Create temp directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / "temp.tiff"
            
            # Save uploaded file
            try:
                with temp_file_path.open("wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            finally:
                file.file.close()
            
            # Validate parameters
            if not (1 <= num_beams <= 10):
                raise HTTPException(
                    status_code=400,
                    detail="Number of beams must be between 1 and 10"
                )
            
            if not (100 <= max_new_tokens <= 4000):
                raise HTTPException(
                    status_code=400,
                    detail="Max new tokens must be between 100 and 4000"
                )
            
            if not (0.1 <= temperature <= 1.0):
                raise HTTPException(
                    status_code=400,
                    detail="Temperature must be between 0.1 and 1.0"
                )

            # Process the image
            try:
                processed_image, extracted_text = process_multipage_tiff(
                    str(temp_file_path),
                    task,
                    num_beams,
                    max_new_tokens,
                    temperature
                )
                
                # Convert processed image to base64
                if processed_image:
                    buffered = BytesIO()
                    processed_image.save(buffered, format="PNG")
                    image_base64 = base64.b64encode(buffered.getvalue()).decode()
                else:
                    image_base64 = None

                return JSONResponse({
                    "success": True,
                    "image_base64": image_base64,
                    "text": extracted_text
                })

            except Exception as processing_error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing image: {str(processing_error)}"
                )

    except HTTPException as http_error:
        return JSONResponse({
            "success": False,
            "error": http_error.detail
        }, status_code=http_error.status_code)
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }, status_code=500)
