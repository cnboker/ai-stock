from typing import List

from fastapi import APIRouter, Depends, Form, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict
from sqlmodel import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from models.user import User  # 确保路径正确
from api.deps import get_session
# 导入之前定义的加密工具函数 (假设在 app.core.security 中)
from core.security import admin_required, get_current_user, hash_password, verify_password 

router = APIRouter()

# 1. 定义响应模型，不写 password 字段
class UserOut(BaseModel):
    id: int
    full_name: str
    role: str
    phone: str
    # 允许从 SQLAlchemy 对象直接转换
    model_config = ConfigDict(from_attributes=True)
    
# response_model 使用 List[User] 来返回用户列表
@router.get("/", response_model=List[UserOut])
async def get_all_users(session: AsyncSession = Depends(get_session)):
    """
    获取所有用户信息
    """
    # 执行查询语句
    # 假设你的 User 模型中 role 字段的值是 "ADMIN"
    statement = (
        select(User)
        .where(User.role != "ADMIN") # 核心：排除管理员
        .order_by(User.id.desc()))   # 建议加上排序，方便前端分页

    result = await session.execute(statement)
    
    # 获取所有的 scalar 结果
    users = result.scalars().all()
    
    return users

# --- 1. 创建用户 (注册) ---
@router.post("/create", response_model=User, status_code=status.HTTP_201_CREATED)
async def create_user(user: User, session: AsyncSession = Depends(get_session),_=Depends(admin_required)):
    # 检查手机号是否已存在
    statement = select(User).where(User.phone == user.phone)
    result = await session.execute(statement)
    if result.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="该手机号已被注册")
    
    # 关键：对明文密码进行加密后再存入数据库
    user.password = hash_password(user.password)
    
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


# --- 3. 分配角色 (修改角色) ---
@router.patch("/{user_id}/role")
async def assign_role(
    user_id: int, 
    new_role: str, 
    session: AsyncSession = Depends(get_session)
    ,_=Depends(admin_required)
):
    """
    修改用户角色。在实际生产中，此接口应配合权限校验，仅限 ADMIN 调用。
    """
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="找不到该人员")
    
    user.role = new_role
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return {"msg": f"用户 {user.full_name} 的角色已更新为 {new_role}"}


# --- 4. 禁用/启用用户 (软删除/状态管理) ---
@router.patch("/{user_id}/status")
async def toggle_user_status(
    user_id: int, 
    is_disabled: bool, 
    session: AsyncSession = Depends(get_session)
    ,_=Depends(admin_required)
):
    """
    禁用或启用用户账号：
    - is_disabled: true (停用) / false (启用)
    """
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="找不到该人员")
    
    user.is_disabled = is_disabled
    session.add(user)
    await session.commit()
    await session.refresh(user)
    
    status_msg = "已停用" if is_disabled else "已启用"
    return {"msg": f"用户 {user.full_name} {status_msg}"}

# --- 5. 删除用户 (物理删除) ---
@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int, 
    session: AsyncSession = Depends(get_session)
    ,_=Depends(admin_required)
):
    """
    从数据库中物理删除用户。
    注意：在有业务关联（如已存在巡查单）的情况下，建议使用上面的禁用接口而非删除。
    """
    user = await session.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="找不到该人员")
    
    await session.delete(user)
    await session.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)

class ResetPasswordSchema(BaseModel):
    old_password: str 
    new_password: str 

@router.patch("/reset_password")
async def reset_password(
    data: ResetPasswordSchema,
    current_user: User = Depends(get_current_user), # 获取当前用户
    db: AsyncSession = Depends(get_session)
):
    # 1. 验证原密码是否正确
    if not verify_password(data.old_password, current_user.password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="原密码输入错误"
        )
    
    # 2. 检查新密码是否与旧密码相同
    if data.old_password == data.new_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="新密码不能与原密码相同"
        )

    # 3. 更新密码
    new_hashed_password = hash_password(data.new_password)
    current_user.password = new_hashed_password
    db.add(current_user)
    await db.commit()
    
    return {"code": 200, "message": "密码重置成功"}