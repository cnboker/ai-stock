from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from api.deps import get_session
from models.user import User
from fastapi import status

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "$szsong100@ad#" # 建议放环境变量
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/login/access-token")


# 定义 Token 提取方式（会从 Header 的 Authorization: Bearer <token> 中取）
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/login", auto_error=False)


async def verify_token(token: str = Depends(oauth2_scheme)):
    # 1. 基础存在性校验（如果 oauth2_scheme 设置了 auto_error=False 才有必要）
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未检测到登录凭证",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        # 2. 解码并验证 Token
        # jwt.decode 会自动校验过期时间 (exp) 和签名
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # 3. 提取信息（假设你在 JWT 中存了 sub 或 username）
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Token 无效：缺少用户信息")
            
        # 4. (可选) 这里可以进一步去数据库查询用户是否存在或被禁用
        
        return {"user": username, "role": payload.get("role", "user")}

    except JWTError:
        # 捕获所有 JWT 相关错误（过期、篡改、格式错误等）
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已过期或凭证无效",
            headers={"WWW-Authenticate": "Bearer"},
        )

# app/api/deps.py 或类似文件

async def get_current_user(token: str = Depends(oauth2_scheme), session: AsyncSession = Depends(get_session)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_phone: str = payload.get("sub") # 这里拿到的其实是 189...
        
        # ❌ 错误写法：session.get(User, user_phone) 会默认查主键 ID，导致溢出
        # user = await session.get(User, user_phone) 

        # ✅ 正确写法：明确指定根据 phone 字段查询
        from sqlmodel import select
        statement = select(User).where(User.phone == user_phone)
        result = await session.execute(statement)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
            
        return user
    except Exception as e:
        print(f"Auth Error: {e}")
        raise HTTPException(status_code=401)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=24) # 设置有效期24小时
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def admin_required(current_user: User = Depends(get_current_user)):
    """
    检查当前用户是否拥有管理员角色
    """
    # 假设你的 User 模型中 roles 是个列表或者有 role 字段
    if current_user.role != "ADMIN": 
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="权限不足，仅限管理员操作"
        )
    return current_user

