# FastAPI Entry Point
from fastapi import Depends, FastAPI
from api.endpoints import api_router_public, api_router_private  # 导入聚合后的路由
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.security import verify_token
# for docker
app = FastAPI(title="ai-stock")
# 定义允许访问的源
# 在开发环境下，可以直接写 ["*"]，或者指定你的前端地址
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,            # 允许携带 Cookie
    allow_methods=["*"],               # 允许所有的请求方法 (GET, POST, 等)
    allow_headers=["*"],               # 允许所有的请求头
    allow_origins=["*"],
)

# 1. 挂载公开路由（无验证）
app.include_router(api_router_public, prefix="/api/v1")

# 2. 挂载私有路由（全局验证）
app.include_router(
    api_router_private, 
    prefix="/api/v1", 
    dependencies=[Depends(verify_token)]
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/hi")
async def root():
    return {"message": "Welcome to ai-stock App API"}
