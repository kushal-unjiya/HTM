from fastapi import FastAPI, UploadFile, File, HTTPException
import textract
import uvicorn

app = FastAPI()

@app.post('/convert')
async def convert_pdf_to_text(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file format. Only PDFs are accepted.")

    try:
        # Convert the PDF to text
        text = textract.process(file.file)
        # Return the text as the response
        return {"text": text.decode('utf-8')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
