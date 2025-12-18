from action.action import Action
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from util.markdown_util import MarkdownUtil
from typing import List, Dict

SYSTEM_PROMPT = """
你是一个知识积累助手，你的任务是根据执行成功的任务和任务步骤总结业务或者是概念知识。

# 注意
1. 业务知识以 QA对 的形式总结。

# 输出结构
## 你的思考
你的思考过程

## 总结的知识
请以 JSON 格式返回知识列表，格式如下：
{
  "knowledge": [
    {
      "question": "问题",
      "answer": "答案",
    },
    {
      "question": "问题",
      "answer": "答案",
    },
    {
      "question": "问题",
      "answer": "答案",
    },
    ...
  ]
}
"""

USER_PROMPT = """
# 执行成功的任务
{task}

# 任务步骤
{steps}
""" 

class AccumulateKnowledgeAction(Action):
    def __init__(self, llm: ChatOpenAI):
        super().__init__(name="accumulate_knowledge", description="积累知识", llm=llm)
    
    async def run(self, task: str, steps: str) -> List[Dict[str, str]]:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=USER_PROMPT.format(task=task, steps=steps))
        ]
        response = await self.llm.ainvoke(messages)
        response = MarkdownUtil.extract_json(response.content, "总结的知识")
        result = response.get("knowledge", [])
        return result