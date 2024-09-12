from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pdfminer.high_level import extract_text
from io import BytesIO
import os
import tempfile
from transformers import pipeline
import pytesseract
from PIL import Image
import librosa

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a temporary directory
temp_dir = tempfile.mkdtemp()

# Initialize the Whisper model
try:
    transcriber = pipeline("automatic-speech-recognition", model="distil-whisper/distil-small.en")
except Exception as e:
    print(f"Error initializing the transcriber: {str(e)}")
    transcriber = None

@app.post('/read_file')
async def read_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file part")

    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    file_extension = os.path.splitext(file.filename)[1].lower()
    
    try:
        # Handle PDF files
        if file_extension == '.pdf':
            file_stream = BytesIO(await file.read())
            text = extract_text(file_stream)
            print("Extracted text from PDF:", text)  # Print the extracted text to the terminal
            return JSONResponse(content={'text': text})

        # Handle image files
        elif file_extension in ['.png', '.jpg', '.jpeg']:
            # Create a temporary file for the image
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension, dir=temp_dir) as temp_file:
                temp_file_path = temp_file.name
                # Write the uploaded file content to the temporary file
                temp_file.write(await file.read())

            # Open the image file
            with Image.open(temp_file_path) as img:
                # Use Tesseract to do OCR on the image
                text = pytesseract.image_to_string(img)
                print("Extracted text from image:", text)  # Print the extracted text to the terminal

            # Remove the temporary file
            os.unlink(temp_file_path)
            return JSONResponse(content={'text': text})

        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

    except Exception as e:
        # If an error occurs, ensure the temporary file is removed
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/upload_audio')
async def upload_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file part")

    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    if transcriber is None:
        raise HTTPException(status_code=500, detail="Transcriber is not initialized")

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=temp_dir) as temp_file:
            temp_file_path = temp_file.name
            # Write the uploaded file content to the temporary file
            temp_file.write(await file.read())

        # Load the audio file
        audio, sr = librosa.load(temp_file_path, sr=16000)

        # Transcribe the audio
        result = transcriber(temp_file_path)
        transcription = result["text"]
        print("Transcription:", transcription)  # Print the transcription to the terminal
        # Remove the temporary file
        os.unlink(temp_file_path)

        return JSONResponse(content={'transcription': transcription})
    except Exception as e:
        # If an error occurs, ensure the temporary file is removed
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/record_audio')
async def record_audio(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file part")

    if file.filename == '':
        raise HTTPException(status_code=400, detail="No selected file")

    if transcriber is None:
        raise HTTPException(status_code=500, detail="Transcriber is not initialized")

    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir=temp_dir) as temp_file:
            temp_file_path = temp_file.name
            # Write the uploaded file content to the temporary file
            temp_file.write(await file.read())

        # Load the audio file
        audio, sr = librosa.load(temp_file_path, sr=16000)

        # Transcribe the audio
        result = transcriber(temp_file_path)
        transcription = result["text"]

        # Remove the temporary file
        os.unlink(temp_file_path)

        return JSONResponse(content={'transcription': transcription})
    except Exception as e:
        # If an error occurs, ensure the temporary file is removed
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
