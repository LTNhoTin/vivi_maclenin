import os
import requests
import logging
import base64
import re
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(
        self,
        base_url: str = None,
        model_name: str = "gpt-oss:20b",
        vision_model_name: str = "gemma3:latest",
        response_language: str = "vi",
        max_output_tokens: int = 2048,
        temperature: float = 0.2,
    ):
        if base_url is None:
            base_url = os.getenv("OLLAMA_BASE_URL")
            if not base_url:
                raise ValueError("OLLAMA_BASE_URL environment variable is required")
        if model_name is None:
            model_name = os.getenv("OLLAMA_MODEL_NAME", "gpt-oss:20b")
        if vision_model_name is None:
            vision_model_name = os.getenv("OLLAMA_VISION_MODEL_NAME", "gemma3:latest")
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.vision_model_name = vision_model_name
        self.response_language = response_language
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def build_prompt(self, question: str, contexts: List[Dict]) -> str:
        """Xây dựng prompt theo hai chế độ: có ngữ cảnh (RAG) và không có ngữ cảnh (open-domain)."""
        if contexts and len(contexts) > 0:
            instruction = (
                f"Bạn là ViVi, một chatbot hỗ trợ thông tin dựa trên RAG (Retrieval-Augmented Generation). "
                f"Trả lời bằng tiếng {self.response_language}, giọng tự nhiên và mạch lạc. "
                "Trả lời trực tiếp, rõ ràng và ngắn gọn dựa trên các ngữ cảnh được cung cấp. "
                "Không phân tích hay suy đoán ngoài ngữ cảnh, không mở đầu bằng các cụm như 'Theo ngữ cảnh được cung cấp'. "
                "Không nhắc tới 'Context 1/2' hay 'Source: Context X' trong câu trả lời, và không trích nguồn trừ khi người dùng yêu cầu. "
                "Chỉ dùng bullet khi người dùng yêu cầu; mặc định hãy viết thành một đoạn hoặc vài câu liên kết. "
                "KHÔNG BAO GIỜ tự nhận mình là ChatGPT hay OpenAI. Bạn là ViVi."
            )
            context_texts = []
            for c in contexts:
                txt = c.get("text", "")
                if not txt:
                    continue
                context_texts.append(txt)
            context_block = "\n\n".join(context_texts)
            return f"{instruction}\n\nNgữ cảnh:\n{context_block}\n\nCâu hỏi: {question}"
        else:
            instruction = (
                f"Bạn là ViVi, một chatbot hỗ trợ thông tin dựa trên RAG (Retrieval-Augmented Generation). "
                f"Trả lời bằng tiếng {self.response_language}, giọng thân thiện và ngắn gọn. "
                "Trả lời trực tiếp dựa trên kiến thức chung của bạn. "
                "Nếu câu hỏi yêu cầu thông tin hoặc trích dẫn từ tài liệu cụ thể, hãy nói rằng hiện không có dữ liệu tài liệu để trích dẫn, nhưng vẫn giải thích ngắn gọn theo hiểu biết chung. "
                "Không trích nguồn, không mở đầu bằng các cụm như 'Theo ngữ cảnh được cung cấp'. "
                "KHÔNG BAO GIỜ tự nhận mình là ChatGPT hay OpenAI. Bạn là ViVi."
            )
            return f"{instruction}\n\nCâu hỏi: {question}"

    def answer(
        self, 
        question: str, 
        contexts: List[Dict], 
        image_urls: Optional[List[str]] = None,
        file_urls: Optional[List[str]] = None
    ) -> Tuple[str, Dict]:
        """Trả lời câu hỏi với ngữ cảnh RAG. GPTOSS 20B Finetune chỉ hỗ trợ text, bỏ qua image_urls, file_urls."""
        prompt = self.build_prompt(question, contexts)
        
        # Log system prompt (instruction)
        if contexts and len(contexts) > 0:
            instruction = (
                f"Bạn là ViVi, một chatbot hỗ trợ thông tin dựa trên RAG (Retrieval-Augmented Generation). "
                f"Trả lời bằng tiếng {self.response_language}, giọng tự nhiên và mạch lạc. "
                "Trả lời trực tiếp, rõ ràng và ngắn gọn dựa trên các ngữ cảnh được cung cấp. "
                "Không phân tích hay suy đoán ngoài ngữ cảnh, không mở đầu bằng các cụm như 'Theo ngữ cảnh được cung cấp'. "
                "Không nhắc tới 'Context 1/2' hay 'Source: Context X' trong câu trả lời, và không trích nguồn trừ khi người dùng yêu cầu. "
                "Chỉ dùng bullet khi người dùng yêu cầu; mặc định hãy viết thành một đoạn hoặc vài câu liên kết. "
                "KHÔNG BAO GIỜ tự nhận mình là ChatGPT hay OpenAI. Bạn là ViVi."
            )
        else:
            instruction = (
                f"Bạn là ViVi, một chatbot hỗ trợ thông tin dựa trên RAG (Retrieval-Augmented Generation). "
                f"Trả lời bằng tiếng {self.response_language}, giọng thân thiện và ngắn gọn. "
                "Trả lời trực tiếp dựa trên kiến thức chung của bạn. "
                "Nếu câu hỏi yêu cầu thông tin hoặc trích dẫn từ tài liệu cụ thể, hãy nói rằng hiện không có dữ liệu tài liệu để trích dẫn, nhưng vẫn giải thích ngắn gọn theo hiểu biết chung. "
                "Không trích nguồn, không mở đầu bằng các cụm như 'Theo ngữ cảnh được cung cấp'. "
                "KHÔNG BAO GIỜ tự nhận mình là ChatGPT hay OpenAI. Bạn là ViVi."
            )
        
        logger.info("=" * 80)
        logger.info("[RAG DEBUG] ========== SYSTEM PROMPT ==========")
        logger.info(instruction)
        logger.info("=" * 80)
        
        # Log full prompt (input to model)
        logger.info("[RAG DEBUG] ========== FULL PROMPT (INPUT TO MODEL) ==========")
        logger.info(prompt)
        logger.info("=" * 80)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
            }
        }
        
        try:
            logger.info(f"[GPTOSS 20B FINETUNE] Gửi request đến {self.base_url}/api/generate, model: {self.model_name}")
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120  # Tăng timeout lên 120 giây
            )
            response.raise_for_status()
            result = response.json()
            
            # Kiểm tra nếu có lỗi trong response
            if "error" in result:
                error_msg = result.get("error", "Lỗi không xác định từ GPTOSS 20B Finetune")
                logger.error(f"GPTOSS 20B Finetune error: {error_msg}")
                return f"Lỗi GPTOSS 20B Finetune: {error_msg}", {"model": self.model_name, "error": error_msg}
            
            answer = result.get("response", "")
            done_reason = result.get("done_reason", "unknown")
            done = result.get("done", False)
            
            # Log output từ model
            logger.info("[RAG DEBUG] ========== MODEL OUTPUT ==========")
            logger.info(answer)
            logger.info("=" * 80)
            
            # Kiểm tra nếu response bị cắt do giới hạn độ dài
            if done_reason == "length":
                if answer and answer.strip():
                    # Vẫn có nội dung, chỉ bị cắt ngắn - trả về phần đã có kèm cảnh báo
                    logger.warning(f"GPTOSS 20B Finetune response bị cắt do đạt giới hạn max_output_tokens ({self.max_output_tokens}). Độ dài: {len(answer)}")
                    return answer, {"model": self.model_name, "warning": "truncated", "done_reason": "length"}
                else:
                    # Không có nội dung và bị cắt - có thể do prompt quá dài
                    logger.warning(f"GPTOSS 20B Finetune response rỗng và bị cắt do giới hạn độ dài. done_reason: {done_reason}")
                    return "Câu trả lời bị cắt ngắn do giới hạn độ dài. Vui lòng thử lại với câu hỏi ngắn hơn hoặc giảm số lượng ngữ cảnh.", {"model": self.model_name, "warning": "truncated", "done_reason": "length"}
            
            # Kiểm tra nếu answer rỗng
            if not answer or not answer.strip():
                logger.warning(f"GPTOSS 20B Finetune trả về response rỗng. done_reason: {done_reason}, done: {done}, result keys: {list(result.keys())}")
                
                # Kiểm tra các trường hợp đặc biệt
                if done_reason == "stop" and not answer:
                    return "Xin lỗi, tôi không thể tạo câu trả lời. Vui lòng thử lại với câu hỏi khác.", {"model": self.model_name, "warning": "empty_response"}
                elif not done:
                    return "GPTOSS 20B Finetune đang xử lý nhưng chưa hoàn thành. Vui lòng thử lại sau.", {"model": self.model_name, "warning": "not_done"}
                else:
                    return "Không nhận được phản hồi từ GPTOSS 20B Finetune. Vui lòng thử lại.", {"model": self.model_name, "warning": "empty_response"}
            
            logger.info(f"[GPTOSS 20B FINETUNE] Nhận response thành công, độ dài: {len(answer)} chars")
            return answer, {"model": self.model_name}
        except requests.exceptions.Timeout as e:
            logger.error(f"GPTOSS 20B Finetune timeout: {str(e)}")
            return f"GPTOSS 20B Finetune không phản hồi trong thời gian cho phép (timeout). Vui lòng thử lại.", {"model": self.model_name, "error": "timeout"}
        except requests.exceptions.ConnectionError as e:
            logger.error(f"GPTOSS 20B Finetune connection error: {str(e)}")
            return f"Không thể kết nối đến GPTOSS 20B Finetune server tại {self.base_url}. Vui lòng kiểm tra kết nối mạng hoặc liên hệ quản trị viên.", {"model": self.model_name, "error": "connection_error"}
        except requests.exceptions.RequestException as e:
            logger.error(f"GPTOSS 20B Finetune request error: {str(e)}")
            return f"Lỗi kết nối GPTOSS 20B Finetune: {str(e)}", {"model": self.model_name, "error": str(e)}
        except Exception as e:
            logger.error(f"GPTOSS 20B Finetune unexpected error: {str(e)}", exc_info=True)
            return f"Lỗi gọi GPTOSS 20B Finetune: {str(e)}", {"model": self.model_name, "error": str(e)}
    
    def _extract_base64_from_data_url(self, data_url: str) -> str:
        """Extract base64 string từ data URL (data:image/jpeg;base64,...)"""
        # Pattern: data:image/xxx;base64,<base64_data>
        match = re.search(r'base64,(.+)', data_url)
        if match:
            return match.group(1)
        # Nếu không có prefix, giả sử đã là base64
        return data_url
    
    def analyze_image_with_gemma3(
        self,
        question: str,
        image_urls: List[str]
    ) -> Tuple[str, Dict]:
        """
        Phân tích ảnh bằng Gemma3 model qua GPTOSS 20B Finetune.
        Trả về mô tả/phân tích ảnh từ Gemma3.
        """
        if not image_urls or len(image_urls) == 0:
            return "", {"model": self.vision_model_name, "error": "No images provided"}
        
        # Extract base64 từ data URLs
        base64_images = []
        for img_url in image_urls:
            base64_data = self._extract_base64_from_data_url(img_url)
            base64_images.append(base64_data)
        
        # Chỉ lấy ảnh đầu tiên (GPTOSS 20B Finetune có thể hỗ trợ nhiều ảnh nhưng để đơn giản dùng 1 ảnh)
        base64_image = base64_images[0]
        
        # Tạo prompt cho Gemma3
        prompt = (
            f"Hãy mô tả chi tiết hình ảnh này bằng tiếng {self.response_language}. "
            f"Nếu có câu hỏi: '{question}', hãy trả lời dựa trên nội dung trong ảnh. "
            "Mô tả rõ ràng, chi tiết những gì bạn thấy trong ảnh."
        )
        
        # GPTOSS 20B Finetune /api/chat format cho vision
        payload = {
            "model": self.vision_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64_image]
                }
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_output_tokens,
            }
        }
        
        try:
            logger.debug(f"Gửi request đến GPTOSS 20B Finetune vision: {self.base_url}/api/chat, model: {self.vision_model_name}")
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=180  # Vision model cần thời gian xử lý lâu hơn
            )
            response.raise_for_status()
            result = response.json()
            
            # Kiểm tra lỗi
            if "error" in result:
                error_msg = result.get("error", "Lỗi không xác định từ GPTOSS 20B Finetune")
                logger.error(f"GPTOSS 20B Finetune vision error: {error_msg}")
                return "", {"model": self.vision_model_name, "error": error_msg}
            
            # Lấy message từ response
            message = result.get("message", {})
            image_analysis = message.get("content", "")
            
            if not image_analysis or not image_analysis.strip():
                logger.warning(f"Gemma3 trả về phân tích rỗng")
                return "", {"model": self.vision_model_name, "warning": "empty_response"}
            
            logger.debug(f"Gemma3 trả về phân tích ảnh thành công, độ dài: {len(image_analysis)}")
            return image_analysis, {"model": self.vision_model_name}
            
        except requests.exceptions.Timeout as e:
            logger.error(f"GPTOSS 20B Finetune vision timeout: {str(e)}")
            return "", {"model": self.vision_model_name, "error": "timeout"}
        except requests.exceptions.ConnectionError as e:
            logger.error(f"GPTOSS 20B Finetune vision connection error: {str(e)}")
            return "", {"model": self.vision_model_name, "error": "connection_error"}
        except requests.exceptions.RequestException as e:
            logger.error(f"GPTOSS 20B Finetune vision request error: {str(e)}")
            return "", {"model": self.vision_model_name, "error": str(e)}
        except Exception as e:
            logger.error(f"GPTOSS 20B Finetune vision unexpected error: {str(e)}", exc_info=True)
            return "", {"model": self.vision_model_name, "error": str(e)}
