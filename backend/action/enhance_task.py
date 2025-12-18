from action.action import Action
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from util.markdown_util import MarkdownUtil

SYSTEM_PROMPT = """
根据背景知识优化任务中的模糊的概念描述，注意不要增加新概念或行为。

# 注意
1. 使用跟原始任务匹配的背景知识，背景知识不一定要用上。

# 输出结构
## 思考
你的思考过程

## 优化后的任务描述
优化后的任务描述
"""

USER_PROMPT = """
背景知识
{background_knowledge}

原始任务描述
{original_task}
"""


class EnhanceTaskAction(Action):
    def __init__(self, llm: ChatOpenAI):
        super().__init__(name="enhance_task", description="增强任务", llm=llm)
    
    async def run(self, background_knowledge: str, original_task: str) -> str:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT.format(background_knowledge=background_knowledge, original_task=original_task))
        ]
        response = await self.llm.ainvoke(messages)
        enhanced_task = MarkdownUtil.extract_section(response.content, "优化后的任务描述")
        return enhanced_task