"""
多模态 Action 基类
支持图片输入的多模态 Action，继承自 Action
"""
import base64
from typing import Any, Optional, Union, List, Dict
from action.action import Action
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


class MultimodalAction(Action):
    """
    支持图片的多模态 Action 基类
    
    提供图片处理功能，支持：
    - Base64 编码的图片
    - 截图工具生成的图片
    """
    
    def __init__(
        self,
        name: str = "",
        description: str = "",
        llm: Optional[ChatOpenAI] = None
    ):
        """
        初始化多模态 Action
        
        Args:
            name: Action 名称
            description: Action 描述
            llm: LLM 实例（支持多模态的模型）
        """
        super().__init__(name=name, description=description, llm=llm)
    
    def _is_base64(self, data: str) -> bool:
        """判断是否为 base64 编码的图片"""
        # 检查是否包含 data URI 前缀
        if data.startswith("data:image/"):
            return True
        # 检查是否为纯 base64 字符串（简单判断）
        try:
            base64.b64decode(data, validate=True)
            return len(data) > 100  # base64 图片通常较长
        except:
            return False
    
    def _image_to_base64_data_uri(
        self,
        image_source: Union[str, bytes],
        mime_type: Optional[str] = None
    ) -> str:
        """
        将图片转换为 base64 data URI 格式
        
        Args:
            image_source: 图片源（base64 字符串或字节数据）
            mime_type: MIME 类型（如果为 None 则自动推断）
        
        Returns:
            base64 data URI 字符串，格式: data:image/png;base64,...
        """
        # 如果已经是 data URI 格式，直接返回
        if isinstance(image_source, str) and image_source.startswith("data:image/"):
            return image_source
        
        # 如果是 base64 字符串（无前缀），添加前缀
        if isinstance(image_source, str) and self._is_base64(image_source):
            if not image_source.startswith("data:image/"):
                mime_type = mime_type or "image/png"
                return f"data:{mime_type};base64,{image_source}"
            return image_source
        
        # 如果是字节数据
        if isinstance(image_source, bytes):
            image_bytes = image_source
            mime_type = mime_type or "image/png"
        else:
            raise ValueError(f"不支持的图片源类型: {type(image_source)}")
        
        # 转换为 base64
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{base64_data}"
    
    def _create_multimodal_content(
        self,
        text: Optional[str] = None,
        images: Optional[List[Union[str, bytes]]] = None
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        创建多模态消息内容
        
        Args:
            text: 文本内容
            images: 图片列表（base64 或字节数据）
        
        Returns:
            多模态内容（字符串或内容列表）
        """
        # 如果没有图片，返回纯文本
        if not images:
            return text or ""
        
        # 构建多模态内容列表
        content_parts = []
        
        # 添加图片
        for image in images:
            image_url = self._image_to_base64_data_uri(image)
            content_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            })
        
        # 添加文本（如果有）
        if text:
            content_parts.append({
                "type": "text",
                "text": text
            })
        
        # 如果只有一张图片且没有文本，返回列表
        # 如果有多张图片或包含文本，返回列表
        return content_parts if len(content_parts) > 0 else ""
    
    def _create_multimodal_message(
        self,
        text: Optional[str] = None,
        images: Optional[List[Union[str, bytes]]] = None
    ) -> HumanMessage:
        """
        创建多模态 HumanMessage
        
        Args:
            text: 文本内容
            images: 图片列表（base64 或字节数据）
        
        Returns:
            HumanMessage 实例
        """
        content = self._create_multimodal_content(text=text, images=images)
        return HumanMessage(content=content)
    
    async def run(self, **kwargs) -> Any:
        """
        执行 Action（子类需要实现）
        
        子类可以重写此方法，使用以下辅助方法处理多模态输入：
        - _create_multimodal_content(): 创建多模态内容
        - _create_multimodal_message(): 创建多模态消息
        - _image_to_base64_data_uri(): 转换图片格式
        
        Args:
            **kwargs: 执行参数，可能包含：
                - text: 文本内容
                - images: 图片列表
                - image: 单张图片（会被转换为列表）
                - 其他自定义参数
        
        Returns:
            执行结果
        """
        raise NotImplementedError("子类必须实现 run 方法")
    
    def _extract_images_from_kwargs(self, kwargs: Dict[str, Any]) -> List[Union[str, bytes]]:
        """
        从 kwargs 中提取图片参数
        
        Args:
            kwargs: 参数字典
        
        Returns:
            图片列表
        """
        images = []
        
        # 检查 images 参数（列表）
        if "images" in kwargs and kwargs["images"]:
            images_list = kwargs["images"]
            if isinstance(images_list, list):
                images.extend(images_list)
            else:
                images.append(images_list)
        
        # 检查 image 参数（单张图片）
        if "image" in kwargs and kwargs["image"]:
            images.append(kwargs["image"])
        
        return images

