from datetime import datetime
from typing import TYPE_CHECKING, Optional
from sqlmodel import Relationship, SQLModel, Field


class User(SQLModel, table=True):
    __tablename__ = "user"

    id: Optional[int] = Field(default=None, primary_key=True)
    full_name: str = Field(title="姓名")
    phone: str = Field(unique=True, index=True, title="电话")  # 手机号作为唯一标识
    role: str = Field(title="角色")  # 例如: admin
    password: str = Field(title="密码") # 建议存储哈希后的密文
    
    # 账号状态与安全
    is_disabled: bool = Field(default=False, title="是否停用") 
    login_retry_count: int = Field(default=0, title="登录失败重试次数")
    openid: Optional[str] = Field(default=None, index=True, title="微信OpenID")
    
    created_at: datetime = Field(default_factory=datetime.now, title="创建日期")
