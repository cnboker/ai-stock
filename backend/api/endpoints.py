from fastapi import APIRouter
from backend.api.v1 import analytics


# 创建两个 router
api_router_public = APIRouter()
api_router_private = APIRouter()

api_router_private.include_router(analytics.router, prefix="/analytics", tags=["Hermes Review APIs"]) 
