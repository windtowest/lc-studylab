# 🐳 Docker 部署指南

本项目使用 Docker Compose 进行容器化部署，包含以下服务：
- **PostgreSQL**: 数据库服务（用于 LangGraph checkpointer）
- **Backend**: FastAPI 后端服务
- **Frontend**: Next.js 前端服务

## 📋 前置要求

- Docker 20.10+
- Docker Compose 2.0+

## 🚀 快速启动

### 1. 配置环境变量

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的配置
nano .env
```

### 2. 启动所有服务

```bash
# 启动所有服务（后台运行）
docker-compose up -d

# 查看日志
docker-compose logs -f

# 查看特定服务的日志
docker-compose logs -f backend
docker-compose logs -f postgres
```

### 3. 访问服务

- **前端**: http://localhost:3000
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **PostgreSQL**: localhost:5432

## 🔧 常用命令

### 服务管理

```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose stop

# 重启服务
docker-compose restart

# 停止并删除容器（保留数据）
docker-compose down

# 停止并删除容器和数据卷（⚠️ 会删除数据库数据）
docker-compose down -v
```

### 查看状态

```bash
# 查看运行中的容器
docker-compose ps

# 查看日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f backend
```

### 重新构建

```bash
# 重新构建并启动
docker-compose up -d --build

# 只重新构建特定服务
docker-compose build backend
docker-compose up -d backend
```

## 🗄️ 数据持久化

### PostgreSQL 数据

数据存储在 Docker Volume: `lc-studylab-postgres-data`

```bash
# 查看 volume 信息
docker volume inspect lc-studylab-postgres-data

# 备份数据库
docker exec lc-studylab-postgres pg_dump -U tianzuolin langchain > backup.sql

# 恢复数据库
docker exec -i lc-studylab-postgres psql -U tianzuolin langchain < backup.sql
```

### 应用数据

- `./backend/data` - 文档、索引等数据
- `./backend/logs` - 应用日志

## 🔍 故障排查

### 查看容器状态

```bash
docker-compose ps
```

### 查看详细日志

```bash
# 所有服务
docker-compose logs -f

# 特定服务
docker-compose logs -f backend
docker-compose logs -f postgres
```

### 进入容器调试

```bash
# 进入 backend 容器
docker exec -it lc-studylab-backend bash

# 进入 postgres 容器
docker exec -it lc-studylab-postgres bash

# 连接数据库
docker exec -it lc-studylab-postgres psql -U tianzuolin -d langchain
```

### 检查网络连接

```bash
# 查看网络
docker network ls

# 查看网络详情
docker network inspect lc-studylab_lc-studylab-network
```

### 健康检查

```bash
# 查看健康状态
docker-compose ps

# 手动测试健康检查
docker exec lc-studylab-postgres pg_isready -U tianzuolin
docker exec lc-studylab-backend python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
```

## 🔄 数据迁移

### 从旧容器迁移数据

如果你之前使用 `simple_agent_demo` 的 PostgreSQL 容器：

```bash
# 1. 从旧容器导出数据
docker exec postgres-tzl pg_dump -U tianzuolin langchain > migration.sql

# 2. 启动新的 lc-studylab 服务
cd lc-studylab
docker-compose up -d postgres

# 3. 等待 PostgreSQL 启动
sleep 5

# 4. 导入数据到新容器
docker exec -i lc-studylab-postgres psql -U tianzuolin langchain < migration.sql

# 5. 验证数据
docker exec lc-studylab-postgres psql -U tianzuolin langchain -c "\dt"
```

## 🛡️ 安全建议

1. **修改默认密码**: 在 `.env` 中修改 `POSTGRES_PASSWORD`
2. **不要提交 .env**: 确保 `.env` 在 `.gitignore` 中
3. **限制端口暴露**: 生产环境可以移除 PostgreSQL 的端口映射
4. **使用 secrets**: 生产环境建议使用 Docker secrets 管理敏感信息

## 📊 监控

### 查看资源使用

```bash
# 查看容器资源使用
docker stats

# 查看特定容器
docker stats lc-studylab-backend lc-studylab-postgres
```

### 查看磁盘使用

```bash
# 查看 volume 大小
docker system df -v

# 清理未使用的资源
docker system prune -a
```

## 🔗 相关链接

- [Docker 官方文档](https://docs.docker.com/)
- [Docker Compose 文档](https://docs.docker.com/compose/)
- [PostgreSQL Docker 镜像](https://hub.docker.com/_/postgres)
