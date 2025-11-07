import os
import requests
from typing import List, Dict, Tuple, Optional


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://server.nhotin.space:11434",
        model_name: str = "gpt-oss:20b",
        response_language: str = "vi",
        max_output_tokens: int = 150,
        temperature: float = 0.2,
    ):
        if base_url is None:
            base_url = os.getenv("OLLAMA_BASE_URL", "http://server.nhotin.space:11434")
        if model_name is None:
            model_name = os.getenv("OLLAMA_MODEL_NAME", "gpt-oss:20b")
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.response_language = response_language
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def build_prompt(self, question: str, contexts: List[Dict]) -> str:
        """Xây dựng prompt theo hai chế độ: có ngữ cảnh (RAG) và không có ngữ cảnh (open-domain)."""
        if contexts and len(contexts) > 0:
            instruction = (
                f"Bạn là Maclenin, một chatbot hỗ trợ thông tin dựa trên RAG (Retrieval-Augmented Generation). "
                f"Trả lời bằng tiếng {self.response_language}, giọng tự nhiên và mạch lạc. "
                "Trả lời trực tiếp, rõ ràng và ngắn gọn dựa trên các ngữ cảnh được cung cấp. "
                "Không phân tích hay suy đoán ngoài ngữ cảnh, không mở đầu bằng các cụm như 'Theo ngữ cảnh được cung cấp'. "
                "Không nhắc tới 'Context 1/2' hay 'Source: Context X' trong câu trả lời, và không trích nguồn trừ khi người dùng yêu cầu. "
                "Chỉ dùng bullet khi người dùng yêu cầu; mặc định hãy viết thành một đoạn hoặc vài câu liên kết. "
                "KHÔNG BAO GIỜ tự nhận mình là ChatGPT hay OpenAI. Bạn là Maclenin."
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
                f"Bạn là Maclenin, một chatbot hỗ trợ thông tin dựa trên RAG (Retrieval-Augmented Generation). "
                f"Trả lời bằng tiếng {self.response_language}, giọng thân thiện và ngắn gọn. "
                "Trả lời trực tiếp dựa trên kiến thức chung của bạn. "
                "Nếu câu hỏi yêu cầu thông tin hoặc trích dẫn từ tài liệu cụ thể, hãy nói rằng hiện không có dữ liệu tài liệu để trích dẫn, nhưng vẫn giải thích ngắn gọn theo hiểu biết chung. "
                "Không trích nguồn, không mở đầu bằng các cụm như 'Theo ngữ cảnh được cung cấp'. "
                "KHÔNG BAO GIỜ tự nhận mình là ChatGPT hay OpenAI. Bạn là Maclenin."
            )
            return f"{instruction}\n\nCâu hỏi: {question}"

    def answer(
        self, 
        question: str, 
        contexts: List[Dict], 
        image_urls: Optional[List[str]] = None,
        file_urls: Optional[List[str]] = None
    ) -> Tuple[str, Dict]:
        """Trả lời câu hỏi với ngữ cảnh RAG. Ollama chỉ hỗ trợ text, bỏ qua image_urls, file_urls."""
        prompt = self.build_prompt(question, contexts)
        
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
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            # Kiểm tra nếu có lỗi trong response
            if "error" in result:
                error_msg = result.get("error", "Lỗi không xác định từ Ollama")
                return f"Lỗi Ollama: {error_msg}", {"model": self.model_name, "error": error_msg}
            
            answer = result.get("response", "")
            
            # Kiểm tra nếu answer rỗng
            if not answer:
                # Có thể do done_reason hoặc các lý do khác
                done_reason = result.get("done_reason", "unknown")
                if done_reason == "stop" and not answer:
                    return "Xin lỗi, tôi không thể tạo câu trả lời. Vui lòng thử lại với câu hỏi khác.", {"model": self.model_name, "warning": "empty_response"}
                return "Không nhận được phản hồi từ Ollama. Vui lòng thử lại.", {"model": self.model_name, "warning": "empty_response"}
            
            return answer, {"model": self.model_name}
        except requests.exceptions.RequestException as e:
            return f"Lỗi kết nối Ollama: {str(e)}", {"model": self.model_name, "error": str(e)}
        except Exception as e:
            return f"Lỗi gọi Ollama: {str(e)}", {"model": self.model_name, "error": str(e)}
