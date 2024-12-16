from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from typing import List
import pandas as pd
import os
import shutil
import io
from models.model import fit


app = FastAPI() # 서버 실행 방법: `uvicorn main:app --reload`
 
ALLOWED_EXTENSIONS = {"csv"}
MAX_FILE_SIZE = 10_000_000  # 10 MB
UPLOAD_DIR = "uploaded_files"

os.makedirs(UPLOAD_DIR, exist_ok=True)
 
def allowed_file(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")

    content = await file.read()
    file.file.seek(0)
    print(f'[INFO] file length: {len(content)}, file type: {type(file)}, content type: {type(content)}')
    print(f'file.file: {file.file}')
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    # file_contents = file.file.read(100)  # 파일의 처음 100바이트만 읽기
    # print(file_contents)
    # file.file.seek(0)

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

    fit(
        df=full_df,
        y_col='Churn'
    )

    try:
        print(f'[INFO] remove uploaded files: {file_location}')
        os.remove(file_location)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'업로드된 파일 삭제 중 오류 발생: {str(e)}')
    
    return JSONResponse(content={"filename": file.filename, "dataframe_preview": dataframe_preview})
    return {"filename": file.filename, "dataframe_preview": dataframe_preview}

@app.get("/test/text/")
async def download_test():
    return FileResponse("./test_text.txt", filename="download_test.txt", media_type="text/plain")
 
@app.get("/test/image/")
async def download_image():
    return FileResponse("./test_image.jpg", filename="download_test_image.jpg", media_type="image/jpeg")
 
@app.get("/test/pdf/")
async def download_pdf():
    return FileResponse("./data/classification_report.pdf", filename="download_test.pdf", media_type="application/pdf")


'''

@app.get("/test/text/bytes/")
async def download_test_bytes():
    with open("./test_text.txt", "rb") as f:
	    # 파일경로가 아닌 메모리에 로드된 파일을 전송하기 위함
        file_content = f.read()
 
    headers = {
        "Content-Disposition": "attachment; filename=download_test.txt",
    }
    return StreamingResponse(io.BytesIO(file_content), headers=headers, media_type="text/plain")
 
@app.get("/test/image/bytes/")
async def download_image_bytes():
    # 외부에서 이미지를 읽어서 가져오는 경우
    # 구글에서 Github 로고 검색함
    response = httpx.get(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/GitHub_Invertocat_Logo.svg/1200px-GitHub_Invertocat_Logo.svg.png"
    )
    file_content = response.content
 
    def generate():
        yield file_content
 
    file_name = "download_test_image.png"
    headers = {
        "Content-Disposition": f"attachment; filename={file_name}",
    }
    return StreamingResponse(generate(), headers=headers, media_type="image/png")
#   제너레이터 지우고 아래와 같이 사용해도 됨
#   return StreamingResponse(io.BytesIO(file_content), headers=headers, media_type="image/png")
 
'''