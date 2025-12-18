from action.multimodal_action.multimodal_action import MultimodalAction
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from typing import Optional, List
from util.markdown_util import MarkdownUtil

SYSTEM_PROMPT = """
你是一个UI自动化测试步骤分析助手。
你需要对比两张截图（之前的截图和当前的截图），分析当前步骤执行后的结果。

请仔细分析：
1. 界面变化：对比两张图片，识别出界面上的变化
2. 步骤执行情况：根据步骤描述，判断步骤是否成功执行
3. 异常情况：如果发现任何异常或错误，请明确指出

# 注意
1. 如果你不能指出明显错误，类似于复制粘贴这种看不出变化的操作，请返回 'YES'。
2. 如果本身遇到设置完后关闭界面这种操作，默认是执行完成的。
"""

USER_PROMPT = """
# 总任务
{task}

# 历史步骤
{history_steps}

# 当前执行的步骤
{current_step}

# 输出结构
## 思考
你的思考过程

## 结果
只能是 'YES' 或 'NO'。

## 原因
步骤执行结果的原因
"""


class AnalyzeStep(MultimodalAction):
    def __init__(self, llm: ChatOpenAI):
        super().__init__(name="analyze_step", description="分析步骤", llm=llm)

    async def run(self, history_steps, current_step, previous_image, current_image, task) -> str:
        """
        分析步骤执行结果
        
        Args:
            previous_image: 执行前的截图
            current_image: 执行后的截图
            step: 步骤描述文本
            
        Returns:
            分析结果文本
        """
        # 构建图片列表（过滤掉 None 值）
        images: List = []
        if previous_image:
            images.append(previous_image)
        if current_image:
            images.append(current_image)
        
        # 构建多模态消息
        human_message = self._create_multimodal_message(
            text=USER_PROMPT.format(history_steps=history_steps, current_step=current_step, task=task),
            images=images
        )
        
        # 构建消息列表
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            human_message
        ]
        
        # 调用 LLM
        response = await self.llm.ainvoke(messages)
        result = MarkdownUtil.extract_section(response.content, "结果")
        result = True if "YES" in result else False
        reason = MarkdownUtil.extract_section(response.content, "原因")
        print(result, reason)
        return result, reason
        