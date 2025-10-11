# llava_server.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import base64
import torch
from llava.model import LLaVA  # adjust import based on your LLaVA install

app = FastAPI()

# Allow CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your LLaVA model once
model = LLaVA.load_model("llava-mini")  # adjust path or model name
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

DESCRIPTION_PROMPT = """
You are an expert visual analyst. Carefully describe the given image in a single, detailed paragraph. 
Focus on the main subject, style, colors, lighting, texture, and overall impression. 
Do NOT mention artist names or style transfer details.
ONLY RETURN THE FINAL DESCRIPTION.
"""

@app.post("/generate_description")
async def generate_description(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Generate description
        output = model.generate([DESCRIPTION_PROMPT, image])
        description = output.text.strip()
        
        return {"description": description}
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
