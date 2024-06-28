from fastapi import Depends, HTTPException, Request
from typing import Callable

async def error_handling_dependency(call_next: Callable):
    try:
        return await call_next()
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
