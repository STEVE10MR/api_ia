from fastapi import APIRouter, Form, File, UploadFile
from utils.dependencies import error_handling_dependency
from controllers import addFaceController,consultFaceController,getFaceController,detectFaceController
router = APIRouter()

@router.get("/face")
async def consultFace(): return await getFaceController()

@router.post("/face/consult")
async def consultFace(imagen: UploadFile = File(...)):return await consultFaceController(imagen)
@router.post("/face/detectFace")
async def consultFace(imagen: UploadFile = File(...)):return await detectFaceController(imagen)
@router.post("/face/addFace")
async def addFace(name: str = Form(...), imagen: UploadFile = File(...)):return await addFaceController(name,imagen)

