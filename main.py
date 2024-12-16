from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from typing import List
import pandas as pd
import os
import shutil
import io

from models.model import predict, fit
from models.model import available_models


app = FastAPI() # 서버 실행 방법: `uvicorn main:app --reload`
 
ALLOWED_EXTENSIONS = {"csv"}
MAX_FILE_SIZE = 10_000_000  # 10 MB
UPLOAD_DIR = "uploaded_files"

os.makedirs(UPLOAD_DIR, exist_ok=True)
 
def allowed_file(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/models", response_model=List[str])
async def get_models():
    return available_models

@app.post("/uploadfile/predict")
async def create_upload_file(file: UploadFile = File(...), model: str = Form(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")

    content = await file.read()
    file.file.seek(0)
    print(f'[INFO] file length: {len(content)}, file type: {type(file)}, content type: {type(content)}')
    print(f'file.file: {file.file}')
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        print(f'[INFO] file location: {file_location}')
        # file.file.seek(0)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)    
    except Exception as e:
        os.remove(file_location)
        raise HTTPException(status_code=500, detail=f"파일 업로드 처리 중 오류 발생: {str(e)}")

    try:
        df = pd.read_csv(file_location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV 파일 처리 중 오류 발생: {str(e)}")
    
    try:
        dataframe_preview = df.head().to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f" Pandas Dataframe -> dict 처리 중 요류 발생: {str(e)}")

    chunk_size = 10000
    chunks = []
    for chunk in pd.read_csv(file_location, chunksize=chunk_size):
        # 여기서 각 청크를 처리 (예: 데이터 필터링, 분석 등)
        print('[INFO] print sample in dataframe')
        print(chunk.head(3))
        chunks.append(chunk)
 
    full_df = pd.concat(chunks, ignore_index=True)

    result = predict(
        df=full_df,
        model_name=model
    )

    try:
        print(f'[INFO] remove uploaded files: {file_location}')
        os.remove(file_location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'업로드된 파일 삭제 중 오류 발생: {str(e)}')
    
    return JSONResponse(content={"Score": result})
    
@app.post("/uploadfile/train")
async def create_upload_file(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")

    content = await file.read()
    file.file.seek(0)
    print(f'[INFO] file length: {len(content)}, file type: {type(file)}, content type: {type(content)}')
    print(f'file.file: {file.file}')
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        print(f'[INFO] file location: {file_location}')
        # file.file.seek(0)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)    
    except Exception as e:
        os.remove(file_location)
        raise HTTPException(status_code=500, detail=f"파일 업로드 처리 중 오류 발생: {str(e)}")

    try:
        df = pd.read_csv(file_location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV 파일 처리 중 오류 발생: {str(e)}")
    
    try:
        dataframe_preview = df.head().to_dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f" Pandas Dataframe -> dict 처리 중 요류 발생: {str(e)}")

    chunk_size = 10000
    chunks = []
    for chunk in pd.read_csv(file_location, chunksize=chunk_size):
        # 여기서 각 청크를 처리 (예: 데이터 필터링, 분석 등)
        print('[INFO] print sample in dataframe')
        print(chunk.head(3))
        chunks.append(chunk)
 
    full_df = pd.concat(chunks, ignore_index=True)

    result = fit(
        df=full_df,
        y_col='Churn'
    )

    try:
        print(f'[INFO] remove uploaded files: {file_location}')
        os.remove(file_location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'업로드된 파일 삭제 중 오류 발생: {str(e)}')
    
    return JSONResponse(content={"filename": file.filename, })

@app.get("/test/text/")
async def download_test():
    return FileResponse("./test_text.txt", filename="download_test.txt", media_type="text/plain")
 
@app.get("/test/image/")
async def download_image():
    return FileResponse("./test_image.jpg", filename="download_test_image.jpg", media_type="image/jpeg")
 
@app.get("/download_pdf/")
async def download_pdf():
    return FileResponse("./models/results/classification_report.pdf", filename="download_test.pdf", media_type="application/pdf")

