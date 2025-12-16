"""
时间相关工具
提供获取当前时间、日期等功能
"""

from datetime import datetime
from langchain_core.tools import tool

from config import get_logger

logger = get_logger(__name__)


@tool
def get_current_time() -> str:
    """
    获取当前时间
    
    返回格式化的当前日期和时间，格式为：YYYY-MM-DD HH:MM:SS
    
    **注意：查询天气时不需要调用此工具！天气工具内部已经知道当前日期。**
    
    Returns:
        当前时间的字符串表示
        
    Example:
        >>> get_current_time()
        '2025-11-05 14:30:25'
    
    使用场景：
        - 当用户明确问"现在几点"、"当前时间"时使用
        - **不要**在查询天气前调用此工具
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.debug(f"获取当前时间: {current_time}")
    return f"当前时间是：{current_time}"


@tool
def get_current_date() -> str:
    """
    获取当前日期
    
    返回格式化的当前日期，格式为：YYYY-MM-DD，以及星期几
    
    **注意：查询天气时不需要调用此工具！天气工具内部已经知道当前日期。**
    
    Returns:
        当前日期的字符串表示
        
    Example:
        >>> get_current_date()
        '2025-11-05 (星期三)'
    
    使用场景：
        - 当用户明确问"今天是几号"、"今天星期几"时使用
        - **不要**在查询天气前调用此工具
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    
    # 中文星期映射
    weekday_map = {
        0: "星期一",
        1: "星期二",
        2: "星期三",
        3: "星期四",
        4: "星期五",
        5: "星期六",
        6: "星期日",
    }
    weekday = weekday_map[now.weekday()]
    
    result = f"{date_str} ({weekday})"
    logger.debug(f"获取当前日期: {result}")
    return f"今天是：{result}"

