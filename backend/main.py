"""
AI Agent 自动化测试框架 - LangGraph 执行图
"""
from math import e
import os
import sys
import asyncio
from typing import TypedDict, Annotated, Literal, List, Dict, Any
from typing_extensions import Optional
from action.enhance_task import EnhanceTaskAction
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from business_knowledge.database import get_db as get_business_db
from reasoning_knowledge.database import get_db as get_reasoning_db
from task_storage.database import get_db as get_task_db
from business_knowledge.crud import BusinessKnowledgeCRUD
import config
from action.decompose_task import DecomposeTaskAction
from action.ui_tars import UITars
from task_storage.database import get_db as get_task_db
from task_storage.crud import TaskStorageCRUD
import json
from action.judgment_task import JudgmentTask
from reasoning_knowledge.crud import ReasoningKnowledgeCRUD
from util.screenshot_util import ScreenshotUtil
from action.multimodal_action.analyze_step import AnalyzeStep
from action.multimodal_action.optimize_step import OptimizeStep
from action.accumulate_knowledge import AccumulateKnowledgeAction

# 配置（从全局配置字典获取）
OPENAI_API_KEY = config.config_dict.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = config.config_dict.get("OPENAI_BASE_URL", "")
MODEL_NAME = config.config_dict.get("MODEL_NAME", "gpt-5.1")

UI_TARS_BASE_URL = config.config_dict.get("UI_TARS_BASE_URL", "")
UI_TARS_API_KEY = config.config_dict.get("UI_TARS_API_KEY", "")
UI_TARS_MODEL = config.config_dict.get("UI_TARS_MODEL", "")

# 初始化 LLM
llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL if OPENAI_BASE_URL else None,
    temperature=0.3,
)

tars = UITars(base_url=UI_TARS_BASE_URL, api_key=UI_TARS_API_KEY, model=UI_TARS_MODEL)

# === 状态定义 ===
class AgentState(TypedDict):
    """Agent 执行状态"""
    # 输入
    original_task: str  # 原始任务描述
    
    # 补全后的任务
    enhanced_task: Optional[str]  # 补全后的任务描述
    
    # 执行判断
    can_execute: Optional[bool]  # 是否能够执行
    execution_reason: Optional[str]  # 不能执行的原因
    
    # 子任务
    steps: List[Dict[str, Any]]  # 拆解后的步骤列表
    current_step_index: int  # 当前执行的步骤索引
    
    # 执行结果
    step_results: List[Dict[str, Any]]  # 步骤执行结果
    final_result: Optional[Dict[str, Any]]  # 最终结果
    
    # 消息历史
    messages: Annotated[List, add_messages]  # 消息历史


# === 节点函数 ===

async def enhance_task_node(state: AgentState) -> AgentState:
    """
    节点1: 补全执行任务
    使用 LLM 补全和优化任务描述
    """
    print(f"[补全任务节点] 原始任务: {state['original_task']}")
    db = next(get_business_db())
    crud = BusinessKnowledgeCRUD(db)
    # 根据问题搜索
    results = crud.search_by_question(
        query_text=state['original_task'],
        top_k=5,
        threshold=0.5
    )
    background_knowledge = "\n".join(["问题：" + result['question_text'] + " 回答：" + result['answer_text'] for result in results])

    action = EnhanceTaskAction(llm)
    enhanced_task = await action.run(background_knowledge=background_knowledge, original_task=state['original_task'])
    
    print(f"[补全任务节点] 补全后任务: {enhanced_task}")

    return {
        **state,
        "enhanced_task": enhanced_task,
        "messages": [AIMessage(content=f"补全后的任务: {enhanced_task}")]
    }


async def check_executability_node(state: AgentState) -> AgentState:
    """
    节点2: 判断是否能够执行
    使用 LLM 判断任务是否可执行
    """
    print(f"[判断可执行性节点] 检查任务: {state['enhanced_task']}")

    db = next(get_task_db())
    crud = TaskStorageCRUD(db)
    results = crud.search_by_enhanced_task(
        query_text=state['enhanced_task'],
        top_k=3,
        threshold=0.8
    )
    if results:
        history_tasks = "\n".join([result['enhanced_task']  + ": 可以执行" if result['all_success'] else result['enhanced_task']  + ": 不能执行" + result['execution_reason']  for result in results])
    else:
        history_tasks = ""
    
    action = JudgmentTask(llm)
    can_execute, execution_reason = await action.run(history_tasks=history_tasks, task=state['enhanced_task'])
    
    return {
        **state,
        "can_execute": can_execute,
        "execution_reason": execution_reason,
        "messages": [AIMessage(content=f"可执行性判断: {'可执行' if can_execute else '不可执行'} - {execution_reason}")]
    }


async def decompose_task_node(state: AgentState) -> AgentState:
    """
    节点3: 拆解子任务
    将任务拆解成多个可执行的步骤
    """
    print(f"[拆解任务节点] 拆解任务: {state['enhanced_task']}")

    db = next(get_reasoning_db())
    crud = ReasoningKnowledgeCRUD(db)
    results = crud.search_by_task(
        query_text=state['enhanced_task'],
        top_k=2,
        threshold=0.8
    )
    if results:
        history_tasks = "\n\n".join([result['task_text'] + "\n" + result['step_text'] for result in results])
    else:
        history_tasks = ""

    db = next(get_business_db())
    crud = BusinessKnowledgeCRUD(db)
    results = crud.search_by_question(
        query_text=state['enhanced_task'],
        top_k=5,
        threshold=0.5
    )
    background_knowledge = "\n".join(["问题：" + result['question_text'] + " 回答：" + result['answer_text'] for result in results])
    action = DecomposeTaskAction(llm)
    steps = await action.run(history_tasks=history_tasks, background_knowledge=background_knowledge, task=state['enhanced_task'])
    print(f"[拆解任务节点] 拆解出 {len(steps)} 个步骤: {steps}")
    
    return {
        **state,
        "steps": steps,
        "current_step_index": 0,
        "step_results": [],
        "messages": [AIMessage(content=f"拆解出 {len(steps)} 个步骤")]
    }


async def execute_subtask_node(state: AgentState) -> AgentState:
    """
    节点4: 执行子任务
    """
    current_step_index = state.get("current_step_index", 0)
    steps = state.get("steps", [])
    # 创建 step_results 的副本，避免直接修改原始状态
    step_results = state.get("step_results", [])[:]
    
    print(f"[执行子任务节点] 当前状态 - 索引: {current_step_index}, 总步骤数: {len(steps)}, 已完成: {len(step_results)}")
    
    if current_step_index >= len(steps):
        print(f"[执行子任务节点] 所有子任务已完成")
        return state
    
    current_step = steps[current_step_index]
    print(f"[执行子任务节点] 执行子任务 {current_step_index + 1}/{len(steps)}: {current_step.get('step', '')}")

    task = state.get("enhanced_task", "")
    history_steps = "\n".join([f"{idx+1}. {step['step_description']}" for idx, step in enumerate(step_results)])
    need_execute_step = current_step['step']

    analyze_action = AnalyzeStep(llm)
    optimize_action = OptimizeStep(llm)

    try:
        n = 0
        first_flag = False
        second_flag = False
        is_first_attempt_success = False  # 标记是否第一次就成功
        while n < 3:
            try:
                previous_image_bytes = ScreenshotUtil.capture_full_screen_bytes()
                
                instruction = """
                你需要执行'现在需要执行的步骤'中的步骤。

                # 现在需要执行的步骤
                {need_execute_step}

                # 注意事项
                可根据实际情况调整'现在需要执行的步骤'，但要保证调整后的步骤所做的事与'现在需要执行的步骤'一致。
                """

                result = await tars.run(instruction=instruction.format(need_execute_step=need_execute_step))
                first_flag = result.get('success', False)
                print(f"[执行子任务节点] 执行结果: {first_flag}")
                if first_flag:
                    current_image_bytes = ScreenshotUtil.capture_full_screen_bytes()
                    second_flag, reason = await analyze_action.run(
                        task=state.get("enhanced_task"),
                        history_steps=history_steps,
                        current_step=need_execute_step,
                        previous_image=previous_image_bytes,
                        current_image=current_image_bytes
                    )
                    if second_flag:
                        # 如果第一次就成功（n == 0），标记为完美步骤
                        if n == 0:
                            is_first_attempt_success = True
                        break
                    else:
                        # 优化步骤
                        n += 1
                        print(f"[执行子任务节点] 分析失败，开始优化步骤 (第 {n} 次)")
                        
                        if n >= 3:
                            print(f"[执行子任务节点] 优化次数超过3次，标记为失败并结束")
                            # 标记当前步骤为失败
                            step_result = {
                                "step_id": current_step.get("id"),
                                "step_description": current_step.get("step"),
                                "success": False,
                                "error": "优化步骤超过3次，自动结束",
                                "analysis": reason if 'reason' in locals() else "分析失败",
                            }
                            step_results.append(step_result)
                            next_index = current_step_index + 1
                            return {
                                **state,
                                "steps": steps,  # 更新优化后的步骤
                                "current_step_index": next_index,
                                "step_results": step_results,
                                "messages": [AIMessage(content=f"子任务 {current_step_index + 1} 优化失败，已标记为失败")]
                            }
                        
                        # 获取剩余步骤（包含当前步骤）
                        remaining_steps = steps[current_step_index:]
                        # 明确标注当前步骤，确保优化时包含当前步骤
                        remaining_steps_text = "\n".join([
                            f"{idx+1}. {step.get('step', '')}" + (" (当前步骤)" if idx == 0 else "")
                            for idx, step in enumerate(remaining_steps)
                        ])
                        
                        # 调用优化步骤
                        try:
                            optimized_steps = await optimize_action.run(
                                task=state.get("enhanced_task"),
                                history_steps=history_steps,
                                remaining_steps=remaining_steps_text,
                                current_image=current_image_bytes
                            )
                            
                            if optimized_steps and len(optimized_steps) > 0:
                                # 只更新当前步骤和后续步骤，不修改之前的步骤
                                # 创建新的steps列表，保留之前的步骤不变（复制字典避免引用问题）
                                new_steps = [step.copy() for step in steps[:current_step_index]]  # 保留之前的步骤（0 到 current_step_index-1）
                                
                                # 更新当前步骤（current_step_index）
                                optimized_first_step = optimized_steps[0]
                                need_execute_step = optimized_first_step.get('step', need_execute_step)
                                
                                # 更新当前步骤，保留其他属性（如id）
                                if current_step_index < len(steps):
                                    updated_current_step = steps[current_step_index].copy()
                                    updated_current_step['step'] = need_execute_step
                                    new_steps.append(updated_current_step)
                                
                                # 更新后续步骤（从 current_step_index+1 开始）
                                # 直接用优化后的步骤替换所有后续步骤，不保留多余的原始步骤
                                for i, opt_step in enumerate(optimized_steps[1:], start=1):
                                    target_index = current_step_index + i
                                    if target_index < len(steps):
                                        # 更新现有步骤，保留其他属性
                                        updated_step = steps[target_index].copy()
                                        updated_step['step'] = opt_step.get('step', steps[target_index].get('step', ''))
                                        new_steps.append(updated_step)
                                    else:
                                        # 如果优化后的步骤数量超过原有步骤，添加新步骤
                                        new_steps.append({
                                            'id': str(target_index + 1),
                                            'step': opt_step.get('step', '')
                                        })
                                
                                # 更新steps列表
                                steps = new_steps
                                
                                updated_end_index = current_step_index + len(optimized_steps) - 1
                                print(f"[执行子任务节点] 步骤已优化: {need_execute_step}")
                                print(f"[执行子任务节点] 已更新步骤索引: {current_step_index} 到 {updated_end_index}，之前的 {current_step_index} 个步骤保持不变，后续步骤已完全替换为优化后的步骤")
                            else:
                                print(f"[执行子任务节点] 优化步骤返回空结果，使用原步骤")
                        except Exception as optimize_error:
                            import traceback
                            print(f"[执行子任务节点] 优化步骤失败: {str(optimize_error)}")
                            print(f"[执行子任务节点] 优化错误详情: {traceback.format_exc()}")
                        
                        continue

                else:
                    n += 1
                    continue
            
            except Exception as e:
                import traceback
                print(f"[执行子任务节点] 执行步骤时发生异常: {str(e)}")
                print(f"[执行子任务节点] 错误详情: {traceback.format_exc()}")
                n += 1
                if n >= 2:
                    break
        
        # 循环结束后，检查 n 是否为 3
        if n == 3:
            # 当 n 为 3 时，直接跳到结束节点，并设置 final_result 为 "客户端存在 BUG"
            print(f"[执行子任务节点] 达到最大尝试次数 3，判定为客户端存在 BUG，直接跳到结束节点")
            final_result = {
                "original_task": state.get("original_task"),
                "enhanced_task": state.get("enhanced_task"),
                "total_steps": len(steps),
                "completed_steps": len(step_results),
                "all_success": False,
                "step_results": step_results,
                "summary": "客户端存在 BUG"
            }
            # 设置 current_step_index 为 len(steps)，这样 should_continue_subtasks 会返回 "finalize"
            return {
                **state,
                "steps": steps,
                "current_step_index": len(steps),
                "step_results": step_results,
                "final_result": final_result,
                "messages": [AIMessage(content="达到最大尝试次数，判定为客户端存在 BUG")]
            }
  
        # 循环结束后，处理结果
        # 检查是否成功执行（first_flag 和 second_flag 都为 True 表示成功）
        if first_flag and second_flag:
            # 如果成功，记录成功结果
            step_result = {
                "step_id": current_step.get("id"),
                "step_description": current_step.get("step"),
                "success": True,
                "analysis": "执行成功",
                "first_flag": first_flag,
                "second_flag": second_flag,
                "is_first_attempt_success": is_first_attempt_success,  # 是否第一次就成功
            }
        else:
            # 如果失败，记录失败结果
            step_result = {
                "step_id": current_step.get("id"),
                "step_description": current_step.get("step"),
                "success": False,
                "error": "执行失败或超过最大尝试次数",
                "analysis": "执行失败",
                "first_flag": first_flag,
                "second_flag": second_flag,
                "is_first_attempt_success": False,
            }
        
        step_results.append(step_result)
        # 移动到下一个子任务
        next_index = current_step_index + 1
        print(f"[执行子任务节点] 子任务 {current_step_index + 1} 执行完成，准备执行下一个子任务（索引: {next_index}）")
        print(f"[执行子任务节点] 状态更新: current_step_index {current_step_index} -> {next_index}, step_results 长度: {len(step_results)}")
        
        return {
            **state,
            "steps": steps,  # 更新优化后的步骤
            "current_step_index": next_index,
            "step_results": step_results,
            "messages": [AIMessage(content=f"子任务 {current_step_index + 1} 执行完成")]
        }
    
    except Exception as e:
        import traceback
        print(f"[执行子任务节点] 子任务 {current_step_index + 1} 执行失败: {str(e)}")
        print(f"[执行子任务节点] 错误详情: {traceback.format_exc()}")
        
        # 初始化变量
        previous_image_bytes = None
        current_image_bytes = None
        analysis_result = None
        
        # 如果执行失败，仍然尝试截图和分析
        try:
            previous_image_bytes = ScreenshotUtil.capture_full_screen_bytes()
        except:
            pass
        
        if previous_image_bytes:
            try:
                current_image_bytes = ScreenshotUtil.capture_full_screen_bytes()
                
                # 尝试分析
                task = state.get("enhanced_task", "")
                history_steps = "\n".join([f"{idx+1}. {step['step_description']}" for idx, step in enumerate(step_results)])
                need_execute_step = current_step['step']
                
                try:
                    analysis_result = await analyze_action.run(
                        history_steps=history_steps,
                        current_step=need_execute_step,
                        previous_image=previous_image_bytes,
                        current_image=current_image_bytes
                    )
                except:
                    analysis_result = None
            except:
                pass
        
        step_result = {
            "step_id": current_step.get("id"),
            "step_description": current_step.get("step"),
            "success": False,
            "error": str(e),
            "analysis": analysis_result,
        }
        step_results.append(step_result)
        # 移动到下一个子任务（即使失败也继续）
        next_index = current_step_index + 1
    
    return {
        **state,
        "steps": steps,  # 更新优化后的步骤
        "current_step_index": next_index,
        "step_results": step_results,
        "messages": [AIMessage(content=f"子任务 {current_step_index + 1} 执行完成")]
    }


async def finalize_node(state: AgentState) -> AgentState:
    """
    节点5: 结束节点
    汇总所有执行结果
    """
    print(f"[结束节点] 汇总执行结果")
    
    # 如果 final_result 已经存在（例如在 n == 3 时设置的），则不覆盖
    if state.get("final_result") is not None:
        print(f"[结束节点] 最终结果已存在，直接返回: {state.get('final_result', {}).get('summary', '')}")
        return state
    
    step_results = state.get("step_results", [])
    all_success = all(r.get("success", False) for r in step_results)

    if all_success:
        accumulate_knowledge_action = AccumulateKnowledgeAction(llm)
        task_text = state.get("enhanced_task")
        step_text = "\n".join([
            f"{idx+1}. {step.get('step_description', '')}"
            for idx, step in enumerate(step_results)
        ])
        knowledge_list = await accumulate_knowledge_action.run(task=task_text, steps=step_text)
        print(f"[结束节点] 总结的知识: {knowledge}")
        for knowledge in knowledge_list:
            try:
                db = next(get_reasoning_db())
                crud = ReasoningKnowledgeCRUD(db)
                knowledge = crud.create(
                    task_text=task_text,
                    step_text=step_text
                )
            except Exception as e:
                print(f"[结束节点] 存入知识库失败: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                db.close()

    # 检查所有步骤是否都是第一次就完美执行（first_flag 和 second_flag 都为 True，且 is_first_attempt_success 为 True）
    all_perfect = all(
        r.get("success", False) and 
        r.get("first_flag", False) and 
        r.get("second_flag", False) and 
        r.get("is_first_attempt_success", False)
        for r in step_results
    )



    
    # 如果所有步骤都完美，将任务和步骤存入知识库
    if all_perfect and len(step_results) > 0:
        try:
            db = next(get_reasoning_db())
            crud = ReasoningKnowledgeCRUD(db)
            
            # 准备任务文本和步骤文本
            task_text = state.get("enhanced_task")
            step_text = "\n".join([
                f"{idx+1}. {step.get('step_description', '')}"
                for idx, step in enumerate(step_results)
            ])
            
            # 存入知识库
            knowledge = crud.create(
                task_text=task_text,
                step_text=step_text
            )
            print(f"[结束节点] 完美步骤已存入知识库，ID: {knowledge.id}")
            db.close()
        except Exception as e:
            print(f"[结束节点] 存入知识库失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    final_result = {
        "original_task": state.get("original_task"),
        "enhanced_task": state.get("enhanced_task"),
        "total_steps": len(state.get("steps", [])),
        "completed_steps": len(step_results),
        "all_success": all_success,
        "all_perfect": all_perfect,  # 是否所有步骤都完美
        "step_results": step_results,
        "summary": f"共执行 {len(step_results)} 个步骤，{'全部成功' if all_success else '部分失败'}"
    }
    
    print(f"[结束节点] 最终结果: {final_result['summary']}")
    if all_perfect:
        print(f"[结束节点] 所有步骤都是第一次就完美执行，已存入知识库")
    
    return {
        **state,
        "final_result": final_result,
        "messages": [AIMessage(content=f"任务执行完成: {final_result['summary']}")]
    }


# === 条件边函数 ===

def should_continue_execution(state: AgentState) -> Literal["decompose", "end"]:
    """
    判断是否继续执行：如果可执行则拆解任务，否则结束
    """
    can_execute = state.get("can_execute", False)
    if can_execute:
        return "decompose"
    else:
        return "end"


def should_continue_subtasks(state: AgentState) -> Literal["execute_subtask", "finalize"]:
    """
    判断是否继续执行子任务：如果还有未执行的子任务则继续，否则结束
    """
    current_index = state.get("current_step_index", 0)
    steps = state.get("steps", [])
    
    print(f"[条件判断] 当前索引: {current_index}, 总步骤数: {len(steps)}, 判断结果: {'继续执行' if current_index < len(steps) else '结束'}")
    
    if current_index < len(steps):
        return "execute_subtask"
    else:
        return "finalize"


# === 构建执行图 ===

def create_workflow_graph() -> StateGraph:
    """创建 LangGraph 执行图"""
    
    workflow = StateGraph(AgentState)
    
    # 添加节点
    workflow.add_node("enhance_task", enhance_task_node)
    workflow.add_node("check_executability", check_executability_node)
    workflow.add_node("decompose_task", decompose_task_node)
    workflow.add_node("execute_subtask", execute_subtask_node)
    workflow.add_node("finalize", finalize_node)
    
    # 设置入口点
    workflow.set_entry_point("enhance_task")
    
    # 添加边
    workflow.add_edge("enhance_task", "check_executability")
    workflow.add_conditional_edges(
        "check_executability",
        should_continue_execution,
        {
            "decompose": "decompose_task",
            "end": "finalize"
        }
    )
    workflow.add_edge("decompose_task", "execute_subtask")
    workflow.add_conditional_edges(
        "execute_subtask",
        should_continue_subtasks,
        {
            "execute_subtask": "execute_subtask",  # 循环执行子任务
            "finalize": "finalize"
        }
    )
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


# === 主函数 ===

async def run_task(task: str, task_id: Optional[int] = None) -> Dict[str, Any]:
    """
    执行一个任务
    
    Args:
        task: 任务描述
        task_id: 任务 ID（如果已存在记录，则更新；否则创建新记录）
        
    Returns:
        执行结果，包含 task_id
    """

    # 创建执行图
    app = create_workflow_graph()
    
    # 初始化状态
    initial_state: AgentState = {
        "original_task": task,
        "enhanced_task": None,
        "can_execute": None,
        "execution_reason": None,
        "steps": [],
        "current_step_index": 0,
        "step_results": [],
        "final_result": None,
        "messages": [],
    }
    
    # 执行图
    print(f"\n{'='*60}")
    print(f"开始执行任务: {task}")
    print(f"{'='*60}\n")
    
    
    final_state = await app.ainvoke(initial_state)
    
    print(f"\n{'='*60}")
    print(f"任务执行完成")
    print(f"{'='*60}\n")
    
    # 获取最终结果
    final_result = final_state.get("final_result", {})
    
    # 保存到数据库
    db = next(get_task_db())
    crud = TaskStorageCRUD(db)
    
    try:
        # 准备数据
        steps_str = json.dumps(final_state.get("steps", []), ensure_ascii=False, indent=2) if final_state.get("steps") else None
        step_results_json = final_state.get("step_results", [])
        final_result_str = final_result.get("summary", "")
        
        if task_id:
            # 更新现有记录
            updated_task = crud.update(
                task_id=task_id,
                original_task=final_state.get("original_task"),
                enhanced_task=final_state.get("enhanced_task"),
                can_execute=final_state.get("can_execute"),
                execution_reason=final_state.get("execution_reason"),
                steps=steps_str,
                step_results=step_results_json,
                final_result=final_result_str,
                all_success=final_result.get("all_success", False)
            )
            if updated_task:
                final_result["task_id"] = updated_task.id
        else:
            # 创建新记录
            new_task = crud.create(
                original_task=final_state.get("original_task"),
                enhanced_task=final_state.get("enhanced_task"),
                can_execute=final_state.get("can_execute"),
                execution_reason=final_state.get("execution_reason"),
                steps=steps_str,
                step_results=step_results_json,
                final_result=final_result_str,
                all_success=final_result.get("all_success", False)
            )
            final_result["task_id"] = new_task.id
            print(f"[数据库] 任务已保存，ID: {new_task.id}")
    except Exception as e:
        print(f"[数据库] 保存任务失败: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

    return final_result


async def main():
    """主函数"""
    # 示例任务
    example_task = "在IM聊天窗口中使用截图下的提取文本功能，提取文本并发送给对方。"
    
    result = await run_task(example_task)
    
    print("\n执行结果:")
    print(f"原始任务: {result.get('original_task')}")
    print(f"补全任务: {result.get('enhanced_task')}")
    print(f"子任务数: {result.get('total_subtasks')}")
    print(f"完成数: {result.get('completed_subtasks')}")
    print(f"全部成功: {result.get('all_success')}")
    print(f"摘要: {result.get('summary')}")


if __name__ == "__main__":
    asyncio.run(main())

