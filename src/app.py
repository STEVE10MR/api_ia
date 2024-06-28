
from fastapi import FastAPI
from routers import faceRouter
from starlette.middleware import Middleware
import uvicorn
from middlewares import error
from services.faceService import faceServiceInstance
from datetime import datetime

app = FastAPI()
app.add_middleware(error.ErrorHandlingMiddleware)
app.include_router(faceRouter, prefix="/api", tags=["api"])
@app.get("/")
async def read_root():
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return {"message": "Current Now API DISCO : "+str(formatted_datetime)}
