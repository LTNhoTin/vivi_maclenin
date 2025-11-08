import os
import logging
import requests
from typing import List, Dict, Tuple, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        model_name: Optional[str] = None,
        response_language: str = "vi",
        max_output_tokens: int = 2048,
        temperature: float = 0.2,
    ):
        self.client = OpenAI(api_key=api_key)
        self.api_key = api_key
        # Đọc model name từ env nếu không được truyền vào
        if model_name is None:
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-nano")
        self.model_name = model_name
        self.response_language = response_language
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.use_responses_endpoint = model_name.startswith("gpt-4.1")

    def build_prompt(self, question: str, contexts: List[Dict]) -> str:
        """Xây dựng prompt theo hai chế độ: có ngữ cảnh (RAG) và không có ngữ cảnh (open-domain)."""
        if contexts and len(contexts) > 0:
            instruction = (
                f"Bạn là trợ lý AI trả lời bằng tiếng {self.response_language}, giọng tự nhiên và mạch lạc. "
                "Trả lời trực tiếp, rõ ràng và ngắn gọn dựa trên các ngữ cảnh được cung cấp. "
                "Không phân tích hay suy đoán ngoài ngữ cảnh, không mở đầu bằng các cụm như 'Theo ngữ cảnh được cung cấp'. "
                "Không nhắc tới 'Context 1/2' hay 'Source: Context X' trong câu trả lời, và không trích nguồn trừ khi người dùng yêu cầu. "
                "Chỉ dùng bullet khi người dùng yêu cầu; mặc định hãy viết thành một đoạn hoặc vài câu liên kết."
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
                f"Bạn là trợ lý AI trả lời bằng tiếng {self.response_language}, giọng thân thiện và ngắn gọn. "
                "Trả lời trực tiếp dựa trên kiến thức chung của bạn. "
                "Nếu câu hỏi yêu cầu thông tin hoặc trích dẫn từ tài liệu cụ thể, hãy nói rằng hiện không có dữ liệu tài liệu để trích dẫn, nhưng vẫn giải thích ngắn gọn theo hiểu biết chung. "
                "Không trích nguồn, không mở đầu bằng các cụm như 'Theo ngữ cảnh được cung cấp'."
            )
            return f"{instruction}\n\nCâu hỏi: {question}"

    def answer(
        self, 
        question: str, 
        contexts: List[Dict], 
        image_urls: Optional[List[str]] = None,
        file_urls: Optional[List[str]] = None,
        use_websearch: bool = False
    ) -> Tuple[str, Dict]:
        """Trả lời câu hỏi với ngữ cảnh RAG và tùy chọn hình ảnh, file, web search."""
        
        if self.use_responses_endpoint:
            return self._answer_with_responses_endpoint(
                question, contexts, image_urls, file_urls, use_websearch
            )
        
        prompt = self.build_prompt(question, contexts)
        
        messages = [{"role": "user", "content": prompt}]
        
        if image_urls:
            content = [{"type": "text", "text": prompt}]
            for img_url in image_urls:
                content.append({"type": "image_url", "image_url": {"url": img_url}})
            messages = [{"role": "user", "content": content}]
        
        # Thêm tools cho websearch nếu được bật
        create_kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        }
        
        if use_websearch:
            # Thêm websearch tool cho chat/completions endpoint
            # Format: {"type": "function", "function": {...}}
            create_kwargs["tools"] = [{
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Tìm kiếm thông tin trên web để trả lời câu hỏi về thông tin thời sự, giá cả, hoặc các thông tin cần cập nhật mới nhất",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Truy vấn tìm kiếm trên web"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }]
        
        try:
            response = self.client.chat.completions.create(**create_kwargs)
            message = response.choices[0].message
            
            # Xử lý tool calls nếu model gọi function tool
            if message.tool_calls:
                # Kiểm tra xem có phải web_search tool không
                has_web_search = any(
                    tool_call.function.name == "web_search" 
                    for tool_call in message.tool_calls
                )
                
                if has_web_search:
                    # Nếu model gọi web_search tool, fallback sang /v1/responses endpoint
                    # vì web_search không phải function thực sự mà là built-in
                    return self._answer_with_responses_endpoint(
                        question, contexts, image_urls, file_urls, use_websearch=True
                    )
                else:
                    # Nếu là tool khác, lấy content nếu có
                    answer = message.content or "Đang xử lý..."
            else:
                answer = message.content or ""
            
            return answer, {"model": self.model_name}
        except Exception as e:
            return f"Lỗi gọi OpenAI: {str(e)}", {"model": self.model_name, "error": str(e)}
    
    def _answer_with_responses_endpoint(
        self,
        question: str,
        contexts: List[Dict],
        image_urls: Optional[List[str]] = None,
        file_urls: Optional[List[str]] = None,
        use_websearch: bool = False
    ) -> Tuple[str, Dict]:
        """Sử dụng endpoint /v1/responses cho model gpt-4.1 với hỗ trợ web search, file và image."""
        prompt = self.build_prompt(question, contexts)
        
        request_body = {
            "model": self.model_name
        }
        
        if use_websearch:
            # Thử cả hai format để xem cái nào hoạt động
            request_body["tools"] = [{"type": "web_search_preview"}]
        
        if image_urls or file_urls:
            content = [{"type": "input_text", "text": prompt}]
            
            if image_urls:
                for img_url in image_urls:
                    content.append({
                        "type": "input_image",
                        "image_url": img_url
                    })
            
            if file_urls:
                for file_url in file_urls:
                    content.append({
                        "type": "input_file",
                        "file_url": file_url
                    })
            
            request_body["input"] = [{
                "role": "user",
                "content": content
            }]
        else:
            request_body["input"] = prompt
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            logger.info(f"[OPENAI] Gửi request đến model {self.model_name}, use_websearch={use_websearch}")
            
            response = requests.post(
                "https://api.openai.com/v1/responses",
                headers=headers,
                json=request_body,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            logger.info(f"[OPENAI] Nhận response thành công, status: {response.status_code}")
            
            # Xử lý response format mới với output array
            answer = ""
            citations = []
            
            if "output" in result and isinstance(result["output"], list):
                # Tìm message item trong output array
                for output_item in result["output"]:
                    if output_item.get("type") == "message":
                        content_list = output_item.get("content", [])
                        for content_item in content_list:
                            if content_item.get("type") == "output_text":
                                answer = content_item.get("text", "")
                                # Trích xuất citations nếu có
                                annotations = content_item.get("annotations", [])
                                for ann in annotations:
                                    if ann.get("type") == "url_citation":
                                        citations.append({
                                            "url": ann.get("url", ""),
                                            "title": ann.get("title", ""),
                                            "start_index": ann.get("start_index"),
                                            "end_index": ann.get("end_index")
                                        })
                        break
            elif "choices" in result and len(result["choices"]) > 0:
                # Fallback cho format cũ
                answer = result["choices"][0].get("message", {}).get("content", "")
            elif "text" in result:
                answer = result["text"]
            elif isinstance(result, str):
                answer = result
            else:
                answer = str(result)
            
            meta = {
                "model": self.model_name,
                "endpoint": "/v1/responses",
                "use_websearch": use_websearch
            }
            
            if citations:
                meta["citations"] = citations
            
            return answer, meta
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    error_msg = f"{error_msg}: {error_detail}"
                except:
                    error_msg = f"{error_msg}: {e.response.text}"
            return f"Lỗi gọi OpenAI /v1/responses: {error_msg}", {"model": self.model_name, "error": error_msg}
        except Exception as e:
            return f"Lỗi xử lý response: {str(e)}", {"model": self.model_name, "error": str(e)}
