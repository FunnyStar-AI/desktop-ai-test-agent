"""
截图工具类
支持全屏截图和区域截图
"""
import os
import io
import math
from typing import Optional, Tuple
from datetime import datetime
from pathlib import Path

try:
    from PIL import ImageGrab, Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

# 参考 UI-TARS-desktop 的压缩参数
IMAGE_FACTOR = 28
# MAX_PIXELS_V1_0 = 2700 * IMAGE_FACTOR * IMAGE_FACTOR = 2,116,800
# MAX_PIXELS_V1_5 = 16384 * IMAGE_FACTOR * IMAGE_FACTOR = 12,845,056
# MAX_PIXELS_DOUBAO = 5120 * IMAGE_FACTOR * IMAGE_FACTOR = 4,014,080
# 默认使用 V1_0 的压缩比例
DEFAULT_MAX_PIXELS = 2700 * IMAGE_FACTOR * IMAGE_FACTOR  # 2,116,800


class ScreenshotUtil:
    """截图工具类"""
    
    @staticmethod
    def _compress_image(image_bytes: bytes, max_pixels: int = DEFAULT_MAX_PIXELS) -> bytes:
        """
        压缩图片，参考 UI-TARS-desktop 的实现
        使用 JPEG 格式并设置质量以控制文件大小在 150KB 左右
        
        Args:
            image_bytes: 原始图片字节数据
            max_pixels: 最大像素数，超过此值将进行缩放
            
        Returns:
            bytes: 压缩后的 JPEG 格式字节数据
        """
        if not PIL_AVAILABLE:
            # 如果没有 PIL，直接返回原始数据
            return image_bytes
        
        try:
            # 从字节数据创建图片对象
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size
            current_pixels = width * height
            
            # 如果像素数超过限制，进行缩放
            if current_pixels > max_pixels:
                resize_factor = math.sqrt(max_pixels / current_pixels)
                new_width = int(width * resize_factor)
                new_height = int(height * resize_factor)
                
                # 使用高质量重采样算法
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 如果图片是 RGBA 模式（带透明度），需要转换为 RGB 模式才能保存为 JPEG
            if image.mode in ('RGBA', 'LA', 'P'):
                # 创建白色背景
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                rgb_image.paste(image, mask=image.split()[-1] if image.mode in ('RGBA', 'LA') else None)
                image = rgb_image
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为 JPEG 字节数据
            # JPEG 质量范围 0-100，设置为 40 以控制文件大小在 150KB 左右
            # 质量 40: 约 134 KB，质量 35: 约 103 KB
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='JPEG', quality=40, optimize=True)
            img_bytes.seek(0)
            
            return img_bytes.read()
        except Exception as e:
            # 如果压缩失败，返回原始数据
            print(f"[截图工具] 图片压缩失败: {str(e)}，返回原始数据")
            return image_bytes
    
    @staticmethod
    def capture_screen(output_path: Optional[str] = None, 
                      region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """
        截取屏幕
        
        Args:
            output_path: 输出文件路径，如果为 None 则自动生成
            region: 截图区域 (x, y, width, height)，如果为 None 则截取全屏
        
        Returns:
            str: 截图文件的路径
        
        Raises:
            RuntimeError: 如果没有可用的截图库
        """
        # 优先使用 mss（跨平台，性能好）
        if MSS_AVAILABLE:
            return ScreenshotUtil._capture_with_mss(output_path, region)
        elif PIL_AVAILABLE:
            return ScreenshotUtil._capture_with_pil(output_path, region)
        else:
            raise RuntimeError(
                "未找到可用的截图库。请安装 mss 或 Pillow：\n"
                "  pip install mss\n"
                "  或\n"
                "  pip install Pillow"
            )
    
    @staticmethod
    def _capture_with_mss(output_path: Optional[str] = None,
                          region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """使用 mss 库截图"""
        with mss.mss() as sct:
            if region:
                x, y, width, height = region
                monitor = {
                    "top": y,
                    "left": x,
                    "width": width,
                    "height": height
                }
            else:
                # 获取主显示器
                monitor = sct.monitors[1]  # 0 是所有显示器，1 是主显示器
            
            # 截图
            screenshot = sct.grab(monitor)
            
            # 生成输出路径
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"screenshot_{timestamp}.png"
            
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存截图
            mss.tools.to_png(screenshot.rgb, screenshot.size, output=output_path)
            
            return os.path.abspath(output_path)
    
    @staticmethod
    def _capture_with_pil(output_path: Optional[str] = None,
                          region: Optional[Tuple[int, int, int, int]] = None) -> str:
        """使用 PIL 库截图（主要支持 Windows）"""
        if region:
            x, y, width, height = region
            bbox = (x, y, x + width, y + height)
        else:
            bbox = None
        
        # 截图
        screenshot = ImageGrab.grab(bbox=bbox)
        
        # 生成输出路径
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"screenshot_{timestamp}.png"
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else "."
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存截图
        screenshot.save(output_path)
        
        return os.path.abspath(output_path)
    
    @staticmethod
    def capture_full_screen(output_path: Optional[str] = None) -> str:
        """
        截取全屏
        
        Args:
            output_path: 输出文件路径，如果为 None 则自动生成
        
        Returns:
            str: 截图文件的路径
        """
        return ScreenshotUtil.capture_screen(output_path, region=None)
    
    @staticmethod
    def capture_region(x: int, y: int, width: int, height: int,
                       output_path: Optional[str] = None) -> str:
        """
        截取指定区域
        
        Args:
            x: 区域左上角 x 坐标
            y: 区域左上角 y 坐标
            width: 区域宽度
            height: 区域高度
            output_path: 输出文件路径，如果为 None 则自动生成
        
        Returns:
            str: 截图文件的路径
        """
        return ScreenshotUtil.capture_screen(output_path, region=(x, y, width, height))
    
    @staticmethod
    def capture_screen_bytes(region: Optional[Tuple[int, int, int, int]] = None, 
                            max_pixels: int = DEFAULT_MAX_PIXELS) -> bytes:
        """
        截取屏幕并返回字节数据（不保存文件，自动压缩）
        
        Args:
            region: 截图区域 (x, y, width, height)，如果为 None 则截取全屏
            max_pixels: 最大像素数，超过此值将进行缩放（默认使用 V1_0 的压缩比例）
        
        Returns:
            bytes: 压缩后的 JPEG 格式字节数据（质量40，约150KB）
        
        Raises:
            RuntimeError: 如果没有可用的截图库
        """
        # 先获取原始截图
        if MSS_AVAILABLE:
            raw_bytes = ScreenshotUtil._capture_bytes_with_mss_raw(region)
        elif PIL_AVAILABLE:
            raw_bytes = ScreenshotUtil._capture_bytes_with_pil_raw(region)
        else:
            raise RuntimeError(
                "未找到可用的截图库。请安装 mss 或 Pillow：\n"
                "  pip install mss\n"
                "  或\n"
                "  pip install Pillow"
            )
        
        # 压缩图片
        return ScreenshotUtil._compress_image(raw_bytes, max_pixels)
    
    
    @staticmethod
    def capture_full_screen_bytes(max_pixels: int = DEFAULT_MAX_PIXELS) -> bytes:
        """
        截取全屏并返回字节数据（不保存文件，自动压缩）
        
        Args:
            max_pixels: 最大像素数，超过此值将进行缩放（默认使用 V1_0 的压缩比例）
        
        Returns:
            bytes: 压缩后的 JPEG 格式字节数据（质量40，约150KB）
        """
        # 先获取原始截图
        if MSS_AVAILABLE:
            raw_bytes = ScreenshotUtil._capture_bytes_with_mss_raw(region=None)
        elif PIL_AVAILABLE:
            raw_bytes = ScreenshotUtil._capture_bytes_with_pil_raw(region=None)
        else:
            raise RuntimeError(
                "未找到可用的截图库。请安装 mss 或 Pillow：\n"
                "  pip install mss\n"
                "  或\n"
                "  pip install Pillow"
            )
        
        # 压缩图片
        return ScreenshotUtil._compress_image(raw_bytes, max_pixels)
    
    @staticmethod
    def _capture_bytes_with_mss_raw(region: Optional[Tuple[int, int, int, int]] = None) -> bytes:
        """使用 mss 库截图并返回原始字节数据（未压缩）"""
        with mss.mss() as sct:
            if region:
                x, y, width, height = region
                monitor = {
                    "top": y,
                    "left": x,
                    "width": width,
                    "height": height
                }
            else:
                monitor = sct.monitors[1]
            
            screenshot = sct.grab(monitor)
            return mss.tools.to_png(screenshot.rgb, screenshot.size)
    
    @staticmethod
    def _capture_bytes_with_pil_raw(region: Optional[Tuple[int, int, int, int]] = None) -> bytes:
        """使用 PIL 库截图并返回原始字节数据（未压缩）"""
        if region:
            x, y, width, height = region
            bbox = (x, y, x + width, y + height)
        else:
            bbox = None
        
        screenshot = ImageGrab.grab(bbox=bbox)
        img_bytes = io.BytesIO()
        screenshot.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.read()
    
    @staticmethod
    def get_screen_size() -> Tuple[int, int]:
        """
        获取屏幕尺寸
        
        Returns:
            Tuple[int, int]: (width, height)
        """
        if MSS_AVAILABLE:
            with mss.mss() as sct:
                monitor = sct.monitors[1]  # 主显示器
                return monitor["width"], monitor["height"]
        elif PIL_AVAILABLE:
            screenshot = ImageGrab.grab()
            return screenshot.size
        else:
            raise RuntimeError(
                "未找到可用的截图库。请安装 mss 或 Pillow：\n"
                "  pip install mss\n"
                "  或\n"
                "  pip install Pillow"
            )

