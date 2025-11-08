// chatbotService.js
export const sendMessageChatService = async (promptInput, model, imageUrl = null, useWebsearch = false) => {
    const requestBody = {
      question: promptInput,
      top_k: null,
      image_urls: imageUrl ? [imageUrl] : null,
      file_urls: null,
      use_websearch: useWebsearch
    };

    const response = await fetch('/api/query', {
        method: "post",
        body: JSON.stringify(requestBody),
        headers: new Headers({
          "ngrok-skip-browser-warning": "69420",
          "Content-Type": "application/json"
        }),
      });
    
    if (!response.ok) {
      throw new Error("Network response was not ok");
    }
    
    const result = await response.json();
    // Transform response to match expected format
    return {
      result: result.answer || result.result,
      source_documents: result.contexts || [],
      references: result.meta || {}
    };
  };