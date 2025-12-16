#!/usr/bin/env python3
"""
CLI 演示工具
用于在命令行中测试和演示 Agent 功能

这是一个交互式命令行工具，可以：
1. 测试 Agent 的基本对话功能
2. 演示流式输出效果
3. 测试工具调用
4. 快速验证配置是否正确

使用方法：
    python scripts/demo_cli.py
    python scripts/demo_cli.py --mode coding
    python scripts/demo_cli.py --stream
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional, List

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents import create_base_agent
from core.tools import ALL_TOOLS, BASIC_TOOLS
from config import settings, setup_logging, get_logger

# 初始化日志
setup_logging()
logger = get_logger(__name__)


class Colors:
    """终端颜色代码"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_colored(text: str, color: str = Colors.ENDC):
    """打印彩色文本"""
    print(f"{color}{text}{Colors.ENDC}")


def print_banner():
    """打印欢迎横幅"""
    banner = f"""
{Colors.CYAN}{'=' * 70}
{Colors.BOLD}  TZL-StudyLab 智能学习助手 - CLI 演示工具
{Colors.ENDC}{Colors.CYAN}  版本: {settings.app_version}
  模型: {settings.deepseek_model}
{'=' * 70}{Colors.ENDC}
"""
    print(banner)


def print_help():
    """打印帮助信息"""
    help_text = f"""
{Colors.YELLOW}可用命令:{Colors.ENDC}
  {Colors.GREEN}/help{Colors.ENDC}      - 显示此帮助信息
  {Colors.GREEN}/mode{Colors.ENDC}      - 切换 Agent 模式 (default/coding/research/concise/detailed)
  {Colors.GREEN}/stream{Colors.ENDC}    - 切换流式/非流式输出
  {Colors.GREEN}/tools{Colors.ENDC}     - 切换工具启用/禁用
  {Colors.GREEN}/clear{Colors.ENDC}     - 清空对话历史
  {Colors.GREEN}/info{Colors.ENDC}      - 显示当前配置
  {Colors.GREEN}/quit{Colors.ENDC}      - 退出程序

{Colors.YELLOW}快捷测试:{Colors.ENDC}
  {Colors.CYAN}现在几点？{Colors.ENDC}              - 测试时间工具
  {Colors.CYAN}计算 123 + 456{Colors.ENDC}         - 测试计算器工具
  {Colors.CYAN}搜索 LangChain 1.0.3{Colors.ENDC}   - 测试网络搜索（需要 智谱ai API Key）

直接输入消息开始对话！
"""
    print(help_text)


class ChatSession:
    """聊天会话管理"""
    
    def __init__(
        self,
        mode: str = "default",
        streaming: bool = False,
        use_tools: bool = True,
        use_advanced_tools: bool = False,
    ):
        self.mode = mode
        self.streaming = streaming
        self.use_tools = use_tools
        self.use_advanced_tools = use_advanced_tools
        self.chat_history: List = []
        self.agent = None
        
        self._create_agent()
    
    def _create_agent(self):
        """创建或重新创建 Agent"""
        # 选择工具
        if not self.use_tools:
            tools = []
        elif self.use_advanced_tools:
            tools = ALL_TOOLS
        else:
            tools = BASIC_TOOLS
        
        # 创建 Agent
        self.agent = create_base_agent(
            tools=tools,
            prompt_mode=self.mode,
            # streaming=self.streaming,
            # verbose=False,
        )
        
        logger.info(f"Agent 已创建: mode={self.mode}, streaming={self.streaming}, tools={len(tools)}")
    
    def set_mode(self, mode: str):
        """切换模式"""
        self.mode = mode
        self._create_agent()
        print_colored(f"✅ 已切换到 {mode} 模式", Colors.GREEN)
    
    def toggle_streaming(self):
        """切换流式输出"""
        self.streaming = not self.streaming
        self._create_agent()
        status = "启用" if self.streaming else "禁用"
        print_colored(f"✅ 流式输出已{status}", Colors.GREEN)
    
    def toggle_tools(self):
        """切换工具"""
        self.use_tools = not self.use_tools
        self._create_agent()
        status = "启用" if self.use_tools else "禁用"
        print_colored(f"✅ 工具已{status}", Colors.GREEN)
    
    def clear_history(self):
        """清空对话历史"""
        self.chat_history = []
        print_colored("✅ 对话历史已清空", Colors.GREEN)
    
    def show_info(self):
        """显示当前配置"""
        info = f"""
{Colors.CYAN}当前配置:{Colors.ENDC}
  模式: {Colors.YELLOW}{self.mode}{Colors.ENDC}
  流式输出: {Colors.YELLOW}{'是' if self.streaming else '否'}{Colors.ENDC}
  工具: {Colors.YELLOW}{'启用' if self.use_tools else '禁用'}{Colors.ENDC}
  对话历史: {Colors.YELLOW}{len(self.chat_history)} 条消息{Colors.ENDC}
"""
        print(info)
    
    async def chat(self, message: str) -> str:
        """发送消息并获取回复"""
        from langchain_core.messages import HumanMessage, AIMessage
        
        if self.streaming:
            # 流式输出
            print_colored("助手: ", Colors.BLUE, end="")
            
            full_response = ""
            async for chunk in self.agent.astream(
                input_text=message,
                chat_history=self.chat_history,
            ):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            print()  # 换行
            
            # 更新对话历史
            self.chat_history.append(HumanMessage(content=message))
            self.chat_history.append(AIMessage(content=full_response))
            
            return full_response
        else:
            # 非流式输出
            response = await self.agent.ainvoke(
                input_text=message,
                chat_history=self.chat_history,
            )
            
            # 更新对话历史
            self.chat_history.append(HumanMessage(content=message))
            self.chat_history.append(AIMessage(content=response))
            
            return response


async def main():
    """主函数"""
    print_banner()
    
    # 检查配置
    try:
        settings.validate_required_keys()
    except ValueError as e:
        print_colored(f"配置错误: {e}", Colors.RED)
        print_colored("请在 .env 文件中设置 DEEPSEEK_API_KEY", Colors.YELLOW)
        return
    
    # 检查可选功能
    if not settings.zhipu_api_key:
        print_colored("未配置 zhipu_api_key，网络搜索功能将不可用", Colors.YELLOW)
    
    print_help()
    
    # 创建会话
    session = ChatSession(
        mode="default",
        streaming=False,
        use_tools=True,
        use_advanced_tools=bool(settings.zhipu_api_key),
    )
    
    # 主循环
    while True:
        try:
            # 获取用户输入
            print_colored("\你: ", Colors.GREEN, end="")
            user_input = input().strip()
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input.startswith("/"):
                command = user_input.lower()
                
                if command == "/quit" or command == "/exit" or command == "/q":
                    print_colored("\n再见！", Colors.CYAN)
                    break
                
                elif command == "/help" or command == "/h":
                    print_help()
                
                elif command.startswith("/mode"):
                    parts = command.split()
                    if len(parts) > 1:
                        session.set_mode(parts[1])
                    else:
                        print_colored("用法: /mode <模式名>", Colors.YELLOW)
                        print_colored("可用模式: default, coding, research, concise, detailed", Colors.YELLOW)
                
                elif command == "/stream":
                    session.toggle_streaming()
                
                elif command == "/tools":
                    session.toggle_tools()
                
                elif command == "/clear":
                    session.clear_history()
                
                elif command == "/info":
                    session.show_info()
                
                else:
                    print_colored(f"未知命令: {command}", Colors.RED)
                    print_colored("输入 /help 查看可用命令", Colors.YELLOW)
                
                continue
            
            # 处理正常对话
            response = await session.chat(user_input)
            
            if not session.streaming:
                print_colored(f"助手: {response}", Colors.BLUE)
            
            # 更新对话历史（简化版，不保存完整的 LangChain 消息）
            # 在实际应用中，应该保存完整的消息对象
            
        except KeyboardInterrupt:
            print_colored("\n\n检测到 Ctrl+C，正在退出...", Colors.CYAN)
            break
        
        except Exception as e:
            print_colored(f"\n错误: {e}", Colors.RED)
            logger.error(f"CLI 错误: {e}", exc_info=True)


def print_colored(text: str, color: str = Colors.ENDC, end: str = "\n"):
    """打印彩色文本（支持 end 参数）"""
    print(f"{color}{text}{Colors.ENDC}", end=end)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print_colored("\n\n再见！", Colors.CYAN)
    except Exception as e:
        print_colored(f"\n程序错误: {e}", Colors.RED)
        logger.error(f"程序错误: {e}", exc_info=True)

