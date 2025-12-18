from action.multimodal_action.multimodal_action import MultimodalAction
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from util.markdown_util import MarkdownUtil

SYSTEM_PROMPT = """
你是一个UI自动化测试步骤优化助手。
你需要根据当前截图的情况，改善接下去要执行的步骤，使其更符合自动化测试的要求。

# 注意
1. 测试账户已提前登录飞书客户端，所有操作均在一个账户下实现。
2. 不需要有验证和检查先前步骤是否正确，但需要确保每一步骤执行前后客户端都有明显的变化。
3. 不能有模糊的概念和指代，确保每个步骤都能独立执行，类似于‘复制粘贴’有较强逻辑顺序的操作需要放在一个步骤中。
4. 所有消息都发给‘测试账户’ 。
5. 步骤编号应从接下去要执行的步骤的编号开始，且优化后的步骤数量等于接下去要执行的步骤数量。
6. 基于当前截图改善接下去要执行的步骤，并且步骤描述要使 UI-TARS 能够执行。

# 输出结构
## 思考
你的思考过程

## 优化的步骤
请以 JSON 格式返回步骤列表，格式如下：
{
  "steps": [
    {
      "id": "1",
      "step": "步骤描述",
    },
    {
      "id": "2",
      "step": "步骤描述",
    },
    {
      "id": "3",
      "step": "步骤描述",
    },
    ...
  ]
}
"""


USER_PROMPT = """
# 总任务
{task}

# 历史步骤
{history_steps}

# 接下去要执行的步骤
{remaining_steps}
"""



class OptimizeStep(MultimodalAction):
    def __init__(self, llm: ChatOpenAI):
        super().__init__(name="optimize_step", description="优化步骤", llm=llm)

    async def run(self, remaining_steps, current_image, task, history_steps) -> str:
        human_message = self._create_multimodal_message(
            text=USER_PROMPT.format(remaining_steps=remaining_steps,history_steps=history_steps,task=task),
            images=[current_image]
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            human_message
        ]
        response = await self.llm.ainvoke(messages)
        response = MarkdownUtil.extract_json(response.content, "优化的步骤")
        print(response)
        result = response.get("steps", [])
        return result



