"""
FastAPI HTTP 服务器主入口
提供 REST API 接口，支持聊天、RAG、深度研究等功能

这是整个后端服务的入口点，负责：
1. 初始化 FastAPI 应用
2. 注册所有路由
3. 配置中间件
4. 提供健康检查和文档
"""

import sys
from pathlib import Path

# 确保项目根目录（backend）在 Python 路径中
# 这样无论从哪里运行脚本都能正确导入模块
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time

from config import settings, setup_logging, get_logger
from api.routers import chat, rag, workflow, deep_research

# 初始化日志
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理
    
    在应用启动和关闭时执行必要的初始化和清理工作
    """
    # ==================== 启动时 ====================
    logger.info("=" * 60)
    logger.info(f"{settings.app_name} v{settings.app_version} 正在启动...")
    logger.info("=" * 60)
    
    # 验证配置
    try:
        settings.validate_required_keys()
        logger.info("配置验证通过")
    except ValueError as e:
        logger.warning(f"配置警告: {e}")
    
    # 打印配置信息
    logger.info(f"运行环境:")
    logger.info(f"   - 模型: {settings.deepseek_model}")
    logger.info(f"   - API Base: {settings.deepseek_base_url}")
    logger.info(f"   - 调试模式: {settings.debug}")
    logger.info(f"   - 日志级别: {settings.log_level}")
    
    # 检查可选功能
    if settings.zhipu_api_key:
        logger.info("   - 智谱ai搜索: 已启用")
    else:
        logger.info("   - 智谱ai搜索: 未配置")
    
    logger.info("=" * 60)
    logger.info("应用启动完成，准备接收请求")
    logger.info("=" * 60)
    
    yield
    
    # ==================== 关闭时 ====================
    logger.info("=" * 60)
    logger.info("应用正在关闭...")
    logger.info("=" * 60)
    
    # 清理工作流资源（关闭数据库连接）
    try:
        from workflows.study_flow_graph import cleanup_study_flow
        cleanup_study_flow()
    except Exception as e:
        logger.error(f"清理工作流资源时出错: {e}")
    
    logger.info("资源清理完成")


# ==================== 创建 FastAPI 应用 ====================

app = FastAPI(
    title=settings.app_name,
    description="tzl-StudyLab 智能学习 & 研究助手 - 后端 API",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
    openapi_url="/openapi.json",
)


# ==================== 中间件配置 ====================

# CORS 中间件 - 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    记录所有 HTTP 请求的日志
    
    包括：请求方法、路径、耗时、状态码
    """
    start_time = time.time()
    
    # 记录请求
    logger.info(f"{request.method} {request.url.path}")
    
    # 处理请求
    try:
        response = await call_next(request)
        
        # 计算耗时
        process_time = time.time() - start_time
        
        # 记录响应
        logger.info(
            f"{request.method} {request.url.path} "
            f"- {response.status_code} - {process_time:.3f}s"
        )
        
        # 添加响应头
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"{request.method} {request.url.path} "
            f"- 错误: {str(e)} - {process_time:.3f}s"
        )
        raise


# ==================== 异常处理 ====================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    全局异常处理器
    
    捕获所有未处理的异常，返回统一的错误响应
    """
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if settings.debug else "服务器内部错误",
            "path": str(request.url),
        },
    )


# ==================== 路由注册 ====================

# 注册聊天路由
app.include_router(chat.router)

# 注册 RAG 路由
app.include_router(rag.router)

# 注册工作流路由（第 3 阶段）
app.include_router(workflow.router)

# 注册深度研究路由（第 4 阶段）
app.include_router(deep_research.router)


# ==================== 根路径和健康检查 ====================

@app.get("/")
async def root():
    """
    根路径 - 返回 API 基本信息
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "LC-StudyLab 智能学习 & 研究助手 API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """
    健康检查接口
    
    用于监控和负载均衡器检查服务状态
    """
    return {
        "status": "healthy",
        "version": settings.app_version,
        "debug": settings.debug,
    }


@app.get("/info")
async def get_info():
    """
    获取系统信息
    
    返回当前配置和可用功能
    """
    return {
        "app_name": settings.app_name,
        "version": settings.app_version,
        "model": settings.deepseek_model,
        "features": {
            "chat": True,
            "streaming": True,
            "tools": True,
            "web_search": bool(settings.zhipu_api_key),
            "rag": True,  # 第 2 阶段 ✅
            "workflow": True,  # 第 3 阶段 ✅
            "deep_research": False,  # 第 4 阶段
        },
    }


# ==================== 开发服务器启动 ====================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("0以开发模式启动服务器...")
    
    uvicorn.run(
        "api.http_server:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.server_reload,
        log_level=settings.log_level.lower(),
    )

