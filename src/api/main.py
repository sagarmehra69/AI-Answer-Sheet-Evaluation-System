from fastapi import FastAPI

from src.api.routes.evaluation import router as evaluation_router
from src.api.routes.upload import router as upload_router
from src.api.routes.evaluate_image import router as evaluate_image_router


app = FastAPI(title="AI Answer Sheet Evaluation System", version="1.0")
app.include_router(upload_router)
app.include_router(evaluation_router)
app.include_router(evaluate_image_router)


@app.get("/")
def home():
    return {"message": "AI Answer Sheet Evaluation System API Running"}
