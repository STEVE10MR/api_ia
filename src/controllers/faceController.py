# src/controllers/faceController.py
from fastapi import UploadFile, HTTPException
from utils import validParams
from services.faceService import faceServiceInstance

async def addFaceController(name: str, imagen: UploadFile):
    if(not validParams.checkParams(name,imagen)):
        raise HTTPException(status_code=400, detail="Invalid parameters")
    return await faceServiceInstance.add_face(name,imagen)

async def consultFaceController(imagen: UploadFile):
    if(not validParams.checkParams(imagen)):
        raise HTTPException(status_code=400, detail="Invalid parameters")
    return await faceServiceInstance.recognize_face(imagen)

async def detectFaceController(imagen: UploadFile):
    if(not validParams.checkParams(imagen)):
        raise HTTPException(status_code=400, detail="Invalid parameters")
    return await faceServiceInstance.detect_face(imagen)

async def getFaceController():
    return {"message": "Test"}
