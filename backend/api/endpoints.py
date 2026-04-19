from fastapi import APIRouter
from api.v1 import analytics, trading
from api.v1 import tasks


# 创建两个 router
api_router_public = APIRouter()
api_router_private = APIRouter()

api_router_public.include_router(analytics.router, prefix="/analytics", tags=["Audit"])
api_router_public.include_router(tasks.router, prefix="/tasks", tags=["Jobs"])
api_router_public.include_router(trading.router, prefix="/trading", tags=["Execution"])

