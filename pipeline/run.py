# This is a sample Python script.
from fastapi import FastAPI, Response, File, UploadFile, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import process

# Press ⌃F5 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

app = FastAPI(
    openapi_url="/recommendation",
    docs_url="/recommendation/docs",
    redoc_url="/recommendation/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.get("/health")
@app.get("/recommendation/health")
def health_check():
    return Response("healthy", status_code=200)


@app.get("/recommendation/")
async def read_item(user_id: int, returnMetadata: bool = False):
    result = process.run_process(user_id, returnMetadata=returnMetadata)
    return JSONResponse(result, status_code=200)


@app.get("/feature/")
async def read_item(user_id: int):
    result = process.run_feature(user_id)
    return JSONResponse(result, status_code=200)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)
