from fastapi import FastAPI, File, Form, UploadFile, Request, HTTPException, Query
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
from utils import load_detect_face, load_tflite_model, process_image, extract_via_google_gemini


app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# UPLOAD_DIR = "uploaded_images"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}

# Load models on startup
load_detect_face()
load_tflite_model()

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the index page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_image(
    request: Request,
    file: UploadFile = File(...),
    response_type: str = Query("json", enum=["json", "html"])
):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    try:
        # file_path = os.path.join(UPLOAD_DIR, file.filename)
        # with open(file_path, "wb") as buffer:
        #     shutil.copyfileobj(file.file, buffer)
        # with open(file_path, "rb") as image_file:
        #     results = process_image(image_file.read())

        file_bytes = await file.read()
        results = process_image(file_bytes)
        if response_type == "html":
            return templates.TemplateResponse("index.html", {"request": request, "results": results})
        return JSONResponse(content={"message": f"Successfully uploaded {file.filename}", "results": results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_text")
async def analyze_text(
    request: Request,
    query: str = Form(...),
    response_type: str = Query("json", enum=["json", "html"])
):
    try:
        result = await extract_via_google_gemini(query)
        if response_type == "html":
            return templates.TemplateResponse("index.html", {"request": request, "result": result, "query": query})
        return JSONResponse(content={"query": query, "result": result})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

