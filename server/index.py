from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pdfminer.high_level import extract_text
from io import BytesIO
import os
import uuid
import shutil  # Import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/read_pdf')
async def read_pdf(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file part")

    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    try:
        file_stream = BytesIO(await file.read())
        text = extract_text(file_stream)
        print("Extracted text:", text)  # Print the extracted text to the terminal
        return JSONResponse(content={'text': text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.2', port=8000)