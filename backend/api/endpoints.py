from fastapi import APIRouter
from api.v1 import analytics, trading
from api.v1 import tasks


# 创建两个 router
api_router_public = APIRouter()
api_router_private = APIRouter()

api_router_private.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Audit"])
api_router_private.include_router(tasks.router, prefix="/api/v1/tasks", tags=["Jobs"])
api_router_private.include_router(trading.router, prefix="/api/v1/trading", tags=["Execution"])

