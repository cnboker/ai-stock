import os
from typing import AsyncGenerator
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelAsyncSession
from typing import Type, TypeVar, Any
from sqlalchemy import and_, select, or_

# 1. 数据库配置 (使用 asyncpg 异步驱动)
# 如果是在 Docker 容器内运行，使用 'db' 作为主机名
#DATABASE_URL = "postgresql+asyncpg://postgres:mysecret@db/aistock"
# host->docker 内部服务名，port->默认 PostgreSQL 端口，dbname->aistock/password->根据你的环境设置
DEFAULT_DB_URL = "postgresql+asyncpg://postgres:mysecret@127.0.0.1:5432/aistock"
# 2. 从环境变量读取，如果没有则使用默认值
# 在 Docker 容器中，你可以设置环境变量 DATABASE_URL=postgresql+asyncpg://postgres:mysecret@db/aistock
DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL)
# 创建异步引擎
engine = create_async_engine(
    DATABASE_URL, 
    echo=True, 
    future=True,
    pool_pre_ping=True  # 自动检查连接是否存活，防止因 Docker 数据库重启导致的连接丢失
)

# 2. 创建异步 Session 工厂
# 注意：使用 SQLModelAsyncSession 可以更好地兼容 SQLModel 的 select 语法
AsyncSessionLocal = sessionmaker(
    engine, 
    class_=SQLModelAsyncSession, 
    expire_on_commit=False
)

# 3. 统一的依赖项：获取异步 Session
async def get_session() -> AsyncGenerator[SQLModelAsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

get_async_session = get_session


