from fastapi import APIRouter, UploadFile, File
import os

router = APIRouter()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    upload_dir = "data/uploads"

    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    return {"filename": file.filename, "saved_to": file_path}
