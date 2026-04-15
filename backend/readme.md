### 初始化项目,Alembic 需要配置为异步模式才能与 asyncpg 协同工作
sudo docker compose exec aistock alembic init -t async migrations

sudo docker compose exec aistock alembic revision --autogenerate -m "initial_setup"
# 生成记录脚本
sudo docker compose exec aistock alembic revision --autogenerate -m "update_table"

# 同步到数据库
alembic upgrade head
# 强制同步版本号（推荐）
alembic stamp head
### 启动 FastAPI
uvicorn main:app --reload
### 容器内迁移数据
sudo docker compose exec aistock alembic upgrade head
### 解决无法修改问题
sudo chmod 666 migrations/versions/275024859f77_create_initial_tables.py
### sql cmd
```bash

sudo docker exec -it backend-db-1 psql -U postgres -d aistock

# 查看所有数据库
\l 
# 查看当前数据库下的所有表
\dt
# 切换到 aistock 数据库
\c aistock
#在 psql 提示符下，输入 \d 加上表名：
\d dictionary
```

### hotload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

### install postgres

```bash
echo "DB_PASSWORD=mysecret" > .env
sudo docker compose up -d --build
```

# --build 会重新根据 Dockerfile 安装依赖
sudo docker compose up -d --build

# 更改帐号
```bash
# 1. 停止并删除现有容器和数据卷 (这会清空数据库里的所有数据！)
sudo docker compose down -v
# 2. 重新启动
sudo docker compose up -d
```

# 创建postgres容器
```base
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=mysecret \
  -p 5432:5432 \
  postgres:15
```

# 初始化数据库
```bash
./init_database.sh
```

# 导出表数据
```bash
# 导出数据库
sudo docker exec -t backend-db-1 pg_dumpall -c -U postgres > alldb_backup.sql
cat alldb_backup.sql | sudo docker exec -i backend-db-1 psql -U postgres -d aistock

pg_dump -U postgres -d aistock -t user_data_permissions --column-inserts --data-only > data.sql
```

# remote server
```bash
chmod 400 ~/.ssh/aistock.server.pem
ssh -i ~/.ssh/aistock.server.pem ubuntu@106.52.105.72 
```

# publish
```bash
# 1. 将代码同步到服务器 (排除 node_modules 和 venv 等无用目录)
rsync -avz -e "ssh -i ~/.ssh/aistock.server.pem" \
--exclude 'venv' --exclude '__pycache__' --exclude '.git' --exclude 'docker_data' --exclude 'static' \
./ ubuntu@106.52.105.72:/home/ubuntu/my-sport-app/backend

rsync -avz -e "ssh -i ~/.ssh/aistock.server.pem" \
alldb_backup.sql ubuntu@106.52.105.72:/home/ubuntu/my-sport-app/backend

rsync -avz -e "ssh -i ~/.ssh/aistock.server.pem" \
Dockerfile ubuntu@106.52.105.72:/home/ubuntu/my-sport-app/backend

rsync -avz -e "ssh -i ~/.ssh/aistock.server.pem" \
aoruite.top_nginx ubuntu@106.52.105.72:/home/ubuntu/my-sport-app/backend

# 2. 远程一键构建并启动
ssh -i aistock.server.pem ubuntu@106.52.105.72 \
"cd /home/ubuntu/my-sport-app/backend && docker-compose up -d --build"
```