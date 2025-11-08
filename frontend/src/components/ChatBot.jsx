"use client"

import { useState, useRef, useEffect, useCallback } from "react"
import ScaleLoader from "react-spinners/ScaleLoader"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import { faMessage, faTrash, faPlus, faImage, faTimes, faDatabase, faGlobe } from "@fortawesome/free-solid-svg-icons"
import ReactMarkdown from "react-markdown"
import robot_img from "../assets/ic5.png"
import { sendMessageChatService } from "./chatbotService"
import commonQuestionsData from "../db/commonQuestions.json"

function ChatBot(props) {
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const fileInputRef = useRef(null)
  const [timeOfRequest, setTimeOfRequest] = useState(0)
  const [promptInput, setPromptInput] = useState("")
  const [model, setModel] = useState("ViVi_pro")
  const [isLoading, setIsLoad] = useState(false)
  const [isGen, setIsGen] = useState(false)
  const [counter, setCounter] = useState(0)
  const [selectedImage, setSelectedImage] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [lightboxImage, setLightboxImage] = useState(null)
  const [useWebsearch, setUseWebsearch] = useState(false)

  // Không lưu cache nữa - reload trang sẽ mất hết lịch sử
  const defaultChats = {
    default: {
      id: "default",
      title: "Cuộc trò chuyện mới",
      createdAt: new Date(),
      messages: [
        [
          "start",
          [
            "Xin chào! Đây là ViVi, trợ lý đắc lực về MLN của bạn! Bạn muốn tìm kiếm thông tin về điều gì?",
            null,
            null,
          ],
        ],
      ],
    },
  }

  // Khởi tạo state không load từ cache
  const [chats, setChats] = useState(defaultChats)
  const [currentChatId, setCurrentChatId] = useState("default")

  const models = [
    {
      value: "ViVi_pro",
      name: "ViVi Pro",
    },
    {
      value: "ViVi",
      name: "ViVi",
    },
  ]

  const commonQuestions = commonQuestionsData

  // Không lưu cache nữa - đã bỏ useEffect lưu cache

  useEffect(() => {
    scrollToEndChat()
    inputRef.current.focus()
  }, [isLoading, currentChatId])

  useEffect(() => {
    const interval = setInterval(() => {
      setTimeOfRequest((timeOfRequest) => timeOfRequest + 1)
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    let interval = null
    if (isLoading) {
      setCounter(1)
      interval = setInterval(() => {
        setCounter((prevCounter) => {
          if (prevCounter < 30) {
            return prevCounter + 1
          } else {
            clearInterval(interval)
            return prevCounter
          }
        })
      }, 1000)
    } else {
      clearInterval(interval)
    }
    return () => clearInterval(interval)
  }, [isLoading])

  const scrollToEndChat = () => {
    messagesEndRef.current.scrollIntoView({ behavior: "auto" })
  }

  const autoResize = (textarea) => {
    textarea.style.height = "auto"
    textarea.style.height = textarea.scrollHeight + "px"
  }

  const onChangeHandler = (event) => {
    setPromptInput(event.target.value)
    autoResize(event.target)
  }

  const sendMessageChat = async () => {
    if ((promptInput !== "" || selectedImage) && isLoading === false) {
      setTimeOfRequest(0)
      setIsGen(true)
      const userMessage = promptInput || (selectedImage ? "Phân tích ảnh này" : "")
      const imageToSend = selectedImage
      setPromptInput("")
      setSelectedImage(null)
      inputRef.current.style.height = "auto"
      setIsLoad(true)

      setChats((prev) => ({
        ...prev,
        [currentChatId]: {
          ...prev[currentChatId],
          messages: [...prev[currentChatId].messages, ["end", [userMessage, model, imageToSend]]],
        },
      }))

      try {
        const result = await sendMessageChatService(userMessage, model, imageToSend, useWebsearch)

        const currentChat = chats[currentChatId]
        let newTitle = currentChat.title
        if (currentChat.messages.length === 2 && currentChat.title === "Cuộc trò chuyện mới") {
          newTitle = userMessage.length > 30 ? userMessage.slice(0, 30) + "..." : userMessage
        }

        setChats((prev) => ({
          ...prev,
          [currentChatId]: {
            ...prev[currentChatId],
            title: newTitle,
            messages: [
              ...prev[currentChatId].messages,
              ["start", [result.result, result.source_documents, result.references, model]],
            ],
          },
        }))
      } catch (error) {
        console.log(error)
        setChats((prev) => ({
          ...prev,
          [currentChatId]: {
            ...prev[currentChatId],
            messages: [...prev[currentChatId].messages, ["start", ["Lỗi, không thể kết nối với server", null, null]]],
          },
        }))
      } finally {
        setIsLoad(false)
        setIsGen(false)
        inputRef.current.focus()
      }
    }
  }

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      sendMessageChat()
    }
  }

  // Hàm nén ảnh để giảm kích thước request
  const compressImage = (file, maxWidth = 1920, maxHeight = 1920, quality = 0.8) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = (e) => {
        const img = new Image()
        img.onload = () => {
          // Tính toán kích thước mới giữ nguyên tỷ lệ
          let width = img.width
          let height = img.height
          
          if (width > maxWidth || height > maxHeight) {
            if (width > height) {
              height = (height * maxWidth) / width
              width = maxWidth
            } else {
              width = (width * maxHeight) / height
              height = maxHeight
            }
          }
          
          // Tạo canvas để resize và nén
          const canvas = document.createElement('canvas')
          canvas.width = width
          canvas.height = height
          const ctx = canvas.getContext('2d')
          ctx.drawImage(img, 0, 0, width, height)
          
          // Convert sang base64 với quality
          const mimeType = file.type || 'image/jpeg'
          const compressedDataUrl = canvas.toDataURL(mimeType, quality)
          resolve(compressedDataUrl)
        }
        img.onerror = reject
        img.src = e.target.result
      }
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }

  const handleImageSelect = async (file) => {
    // Chỉ chấp nhận các định dạng được OpenAI hỗ trợ: png, jpeg, gif, webp
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/gif', 'image/webp']
    const allowedExtensions = ['png', 'jpg', 'jpeg', 'gif', 'webp']
    
    if (file) {
      const fileExt = file.name?.split('.').pop()?.toLowerCase()
      const isValidType = allowedTypes.includes(file.type) || allowedExtensions.includes(fileExt)
      
      // Reject SVG explicitly
      if (file.type === 'image/svg+xml' || fileExt === 'svg') {
        alert("Định dạng SVG không được hỗ trợ. Vui lòng chọn file ảnh hợp lệ (jpg, png, gif, webp)")
        return
      }
      
      if (isValidType) {
        try {
          // Nén ảnh trước khi hiển thị và gửi
          const compressedImage = await compressImage(file)
          setSelectedImage(compressedImage)
        } catch (error) {
          console.error("Lỗi khi nén ảnh:", error)
          // Fallback: dùng ảnh gốc nếu nén thất bại
          const reader = new FileReader()
          reader.onload = (e) => {
            setSelectedImage(e.target.result)
          }
          reader.readAsDataURL(file)
        }
      } else {
        alert("Vui lòng chọn file ảnh hợp lệ (jpg, png, gif, webp). SVG không được hỗ trợ.")
      }
    }
  }

  const handlePaste = (e) => {
    const items = e.clipboardData?.items
    if (!items) return

    for (let i = 0; i < items.length; i++) {
      const item = items[i]
      if (item.type.startsWith("image/")) {
        e.preventDefault()
        const file = item.getAsFile()
        if (file) {
          handleImageSelect(file)
        }
        break
      }
    }
  }

  const handleFileInputChange = (e) => {
    const file = e.target.files?.[0]
    if (file) {
      handleImageSelect(file)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
    // Chỉ set dragging nếu có file ảnh
    if (e.dataTransfer?.types?.includes("Files")) {
      setIsDragging(true)
    }
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    e.stopPropagation()
    // Chỉ set false nếu rời khỏi container, không phải vào child element
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setIsDragging(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const file = e.dataTransfer.files?.[0]
    if (file) {
      handleImageSelect(file)
    }
  }

  const removeImage = () => {
    setSelectedImage(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
  }

  const handleQuickQuestionClick = (question) => {
    const selectedQuestion = commonQuestions.find((q) => q.question === question)
    if (selectedQuestion) {
      setChats((prev) => ({
        ...prev,
        [currentChatId]: {
          ...prev[currentChatId],
          title: question.length > 30 ? question.slice(0, 30) + "..." : question,
          messages: [
            ...prev[currentChatId].messages,
            ["end", [selectedQuestion.question, model]],
            ["start", [selectedQuestion.result, selectedQuestion.source_documents, selectedQuestion.references, model]],
          ],
        },
      }))
      scrollToEndChat()
    }
  }

  const createNewChat = () => {
    const newChatId = `chat-${Date.now()}`
    setChats((prev) => ({
      ...prev,
      [newChatId]: {
        id: newChatId,
        title: "Cuộc trò chuyện mới",
        createdAt: new Date(),
        messages: [
          [
            "start",
            [
              "Xin chào! Đây là ViVi, trợ lý đắc lực về MLN của bạn! Bạn muốn tìm kiếm thông tin về điều gì?",
              null,
              null,
            ],
          ],
        ],
      },
    }))
    setCurrentChatId(newChatId)
    setPromptInput("")
    setSelectedImage(null)
    setUseWebsearch(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
    inputRef.current.style.height = "auto"
  }

  const deleteChat = (chatId) => {
    if (Object.keys(chats).length === 1) {
      alert("Bạn không thể xóa cuộc trò chuyện duy nhất!")
      return
    }
    setChats((prev) => {
      const newChats = { ...prev }
      delete newChats[chatId]
      return newChats
    })
    if (currentChatId === chatId) {
      const remainingChatIds = Object.keys(chats).filter((id) => id !== chatId)
      setCurrentChatId(remainingChatIds[0] || "default")
    }
  }

  // Hàm reset về chat mặc định (không còn cache nữa)
  const clearAllCache = () => {
    if (window.confirm("Bạn có chắc chắn muốn xóa toàn bộ lịch sử trò chuyện? Hành động này không thể hoàn tác.")) {
      // Reset về chat mặc định
      setChats(defaultChats)
      setCurrentChatId("default")
      alert("Đã xóa toàn bộ lịch sử trò chuyện.")
    }
  }

  const switchChat = (chatId) => {
    setCurrentChatId(chatId)
    setPromptInput("")
    setSelectedImage(null)
    setUseWebsearch(false)
    if (fileInputRef.current) {
      fileInputRef.current.value = ""
    }
    inputRef.current.style.height = "auto"
  }

  const currentChat = chats[currentChatId]
  const dataChat = currentChat?.messages || []

  return (
    <div className="bg-gradient-to-r from-orange-50 to-orange-100 flex flex-col w-full h-full overflow-hidden">
      <style>
        {`
                .chat-bubble-gradient-receive {
                    background: linear-gradient(90deg, #f9c6c6 0%, #ffa98a 100%);
                    color: black;
                }
                .chat-bubble-gradient-send {
                    background: linear-gradient(90deg, #2c9fc3 0%, #2f80ed 100%);
                    color: white;
                }
                .input-primary {
                    border-color: #FFA07A;
                }
                .input-primary:focus {
                    outline: none;
                    border-color: #FF6347;
                    box-shadow: 0 0 5px #FF6347;
                }
                .btn-send {
                    background-color: #f8723c !important; 
                    border-color: #FFA07A !important; 
                }
                .btn-send:hover {
                    background-color: #ff9684 !important; 
                    border-color: #FF6347 !important; 
                }
                .textarea-auto-resize {
                    resize: none;
                    overflow: hidden;
                }
                .chat-item-active {
                    background-color: #fff3e0;
                    border-left: 3px solid #f8723c;
                }
            `}
      </style>

      <div className="lg:hidden p-2 flex justify-center bg-gradient-to-r from-orange-50 to-orange-100 flex-shrink-0">
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="w-3/4 p-2 border rounded-lg shadow-md bg-white"
        >
          {models.map((model) => (
            <option key={model.value} value={model.value}>
              {model.name}
            </option>
          ))}
        </select>
      </div>

      <div className="flex flex-1 gap-3 p-3 overflow-hidden min-h-0">
        <div className="hidden lg:flex flex-col w-64 bg-gray-50 rounded-2xl p-3 shadow-md border border-gray-200 overflow-hidden min-w-0">
          <h2 className="font-bold mb-3 text-sm bg-[linear-gradient(90deg,hsl(var(--s))_0%,hsl(var(--sf))_9%,hsl(var(--pf))_42%,hsl(var(--p))_47%,hsl(var(--a))_100%)] bg-clip-text will-change-auto [-webkit-text-fill-color:transparent] flex-shrink-0">
            Những câu hỏi phổ biến
          </h2>
          <ul className="menu text-xs space-y-2 overflow-y-auto overflow-x-hidden flex-1 pr-2 min-w-0">
            {commonQuestions.map((mess, i) => (
              <li key={i} className="hover:bg-orange-100 rounded-lg p-2 cursor-pointer transition min-w-0">
                <button
                  onClick={() => handleQuickQuestionClick(mess.question)}
                  className="text-left text-gray-700 hover:text-gray-900 break-words w-full min-w-0"
                >
                  <FontAwesomeIcon icon={faMessage} className="mr-2 flex-shrink-0" />
                  <span className="break-words">{mess.question}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>

        <div className="flex flex-col flex-1 overflow-hidden">
          <div
            id="chat-area"
            className="
            text-xs lg:text-sm 
            scrollbar-thin scrollbar-thumb-gray-300 bg-white  
            scrollbar-thumb-rounded-full scrollbar-track-rounded-full
            rounded-3xl border-2 border-orange-200 p-4 lg:p-6 overflow-auto flex-1 shadow-inner"
          >
            {dataChat.map((dataMessages, i) =>
              dataMessages[0] === "start" ? (
                <div className="chat chat-start drop-shadow-md" key={i}>
                  <div className="chat-image avatar">
                    <div className="w-8 lg:w-10 rounded-full border-2 border-blue-500 shadow-md">
                      <img className="scale-150" src={robot_img || "/placeholder.svg"} alt="avatar" />
                    </div>
                  </div>
                  <div className="chat-bubble chat-bubble-gradient-receive break-words max-w-[85%] lg:max-w-[70%] shadow-lg">
                    <ReactMarkdown className="prose prose-sm max-w-none">{dataMessages[1][0]}</ReactMarkdown>
                  </div>
                </div>
              ) : (
                <div className="chat chat-end" key={i}>
                  <div className="chat-bubble shadow-xl chat-bubble-gradient-send max-w-[85%] lg:max-w-[70%]">
                    {dataMessages[1][2] && (
                      <div className="mb-3 -mx-2 -mt-2 first:mt-0">
                        <div className="relative group cursor-pointer overflow-hidden rounded-xl border-2 border-white/40 shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-[1.02] bg-gray-100 flex items-center justify-center">
                          <img
                            src={dataMessages[1][2]}
                            alt="Uploaded"
                            onClick={() => setLightboxImage(dataMessages[1][2])}
                            className="w-full max-w-md max-h-64 lg:max-h-80 object-contain transition-transform duration-300 group-hover:scale-105"
                          />
                          <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors duration-300 flex items-center justify-center">
                            <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 bg-white/90 rounded-full p-2 shadow-lg">
                              <FontAwesomeIcon icon={faImage} className="text-blue-600 text-lg" />
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                    {dataMessages[1][0] && (
                      <div className="whitespace-pre-wrap break-words">{dataMessages[1][0]}</div>
                    )}
                  </div>
                </div>
              ),
            )}
            {isLoading && (
              <div className="chat chat-start">
                <div className="chat-image avatar">
                  <div className="w-8 lg:w-10 rounded-full border-2 border-blue-500">
                    <img src={robot_img || "/placeholder.svg"} alt="avatar" />
                  </div>
                </div>
                <div className="flex justify-start px-4 py-2">
                  <div className="chat-bubble chat-bubble-gradient-receive break-words flex items-center">
                    <ScaleLoader color="#0033ff" loading={true} height={15} />
                    <span className="ml-2">{`${counter}/30s`}</span>{" "}
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div
            className={`bg-gradient-to-r from-orange-50 to-orange-100 p-4 rounded-2xl gap-3 mt-3 flex-shrink-0 transition-all border-2 ${
              isDragging ? "border-orange-400 ring-4 ring-orange-200 ring-offset-2 bg-orange-100" : "border-orange-200"
            }`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {selectedImage && (
              <div className="mb-3 p-3 bg-white rounded-xl shadow-lg border-2 border-orange-200">
                <div className="flex items-start gap-3">
                  <div className="relative flex-shrink-0 group bg-gray-100 rounded-lg border-2 border-orange-300 shadow-md overflow-hidden">
                    <img
                      src={selectedImage}
                      alt="Preview"
                      onClick={() => setLightboxImage(selectedImage)}
                      className="w-24 h-24 lg:w-32 lg:h-32 object-contain cursor-pointer hover:scale-105 transition-transform duration-200"
                    />
                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 rounded-lg transition-colors duration-200 flex items-center justify-center">
                      <div className="opacity-0 group-hover:opacity-100 transition-opacity bg-white/90 rounded-full p-1.5">
                        <FontAwesomeIcon icon={faImage} className="text-orange-600 text-sm" />
                      </div>
                    </div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-semibold text-gray-700 mb-1">Ảnh đã chọn</p>
                    <p className="text-xs text-gray-500 mb-2">Nhấn vào ảnh để xem kích thước đầy đủ</p>
                    <button
                      onClick={removeImage}
                      className="btn btn-xs btn-error text-white hover:bg-red-600 transition"
                    >
                      <FontAwesomeIcon icon={faTimes} className="mr-1" />
                      Xóa ảnh
                    </button>
                  </div>
                </div>
              </div>
            )}
            <div className="flex gap-2 items-end">
              <div className="flex-1 flex gap-2 items-end">
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileInputChange}
                  accept="image/*"
                  className="hidden"
                  id="image-upload"
                />
                <label
                  htmlFor="image-upload"
                  className="btn btn-square btn-outline btn-primary border-orange-400 hover:bg-orange-100 hover:border-orange-500 flex-shrink-0 cursor-pointer transition-all shadow-md hover:shadow-lg h-11 w-11"
                  title="Tải ảnh"
                >
                  <FontAwesomeIcon icon={faImage} className="text-orange-600 text-lg" />
                </label>
                <button
                  onClick={() => setUseWebsearch(!useWebsearch)}
                  className={`btn btn-square btn-outline flex-shrink-0 cursor-pointer transition-all shadow-md hover:shadow-lg h-11 w-11 ${
                    useWebsearch
                      ? "bg-orange-500 border-orange-600 text-white hover:bg-orange-600"
                      : "border-orange-400 hover:bg-orange-100 hover:border-orange-500"
                  }`}
                  title={useWebsearch ? "Tắt Web Search (GPT-4.1 nano)" : "Bật Web Search (GPT-4.1 nano)"}
                >
                  <FontAwesomeIcon icon={faGlobe} className={`text-lg ${useWebsearch ? "text-white" : "text-orange-600"}`} />
                </button>
                <textarea
                  placeholder="Nhập câu hỏi tại đây..."
                  className="flex-1 shadow-lg border-2 focus:outline-none px-4 py-3 rounded-xl input-primary textarea-auto-resize min-h-[44px] bg-white focus:bg-orange-50/50 transition-colors"
                  onChange={onChangeHandler}
                  onKeyDown={handleKeyDown}
                  onPaste={handlePaste}
                  onDragOver={(e) => {
                    // Cho phép drag vào textarea
                    if (e.dataTransfer?.types?.includes("Files")) {
                      e.preventDefault()
                      e.stopPropagation()
                    }
                  }}
                  onDrop={(e) => {
                    e.preventDefault()
                    e.stopPropagation()
                    const file = e.dataTransfer.files?.[0]
                    if (file) {
                      handleImageSelect(file)
                    }
                  }}
                  disabled={isGen}
                  value={promptInput}
                  ref={inputRef}
                  rows="1"
                  style={{ resize: "none", overflow: "hidden", lineHeight: "1.5" }}
                />
              </div>
              <button
                disabled={isGen || (!promptInput && !selectedImage)}
                onClick={sendMessageChat}
                className="drop-shadow-lg rounded-xl btn btn-active btn-primary btn-square btn-send flex-shrink-0 disabled:opacity-50 h-11 w-11 transition-all hover:scale-105 active:scale-95"
                title="Gửi tin nhắn"
              >
                <svg
                  stroke="currentColor"
                  fill="none"
                  strokeWidth="2"
                  viewBox="0 0 24 24"
                  color="white"
                  height="20px"
                  width="20px"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <line x1="22" y1="2" x2="11" y2="13"></line>
                  <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
                </svg>
              </button>
            </div>
            <p className="text-xs mt-3 text-justify text-gray-600">
              <b>Lưu ý: </b>ViVi có thể mắc lỗi. Hãy kiểm tra các thông tin quan trọng! {isDragging && (
                <span className="text-orange-600 font-semibold animate-pulse"> - Thả ảnh vào đây để tải lên</span>
              )}
            </p>
          </div>
        </div>

        <div className="hidden lg:flex flex-col w-64 bg-gray-50 rounded-2xl p-3 shadow-md border border-gray-200 overflow-hidden">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-bold text-sm bg-[linear-gradient(90deg,hsl(var(--s))_0%,hsl(var(--sf))_9%,hsl(var(--pf))_42%,hsl(var(--p))_47%,hsl(var(--a))_100%)] bg-clip-text will-change-auto [-webkit-text-fill-color:transparent]">
              Lịch sử trò chuyện
            </h2>
            <div className="flex items-center gap-2">
              <button
                onClick={clearAllCache}
                className="text-red-500 hover:text-red-700 transition"
                title="Xóa toàn bộ lịch sử trò chuyện"
              >
                <FontAwesomeIcon icon={faDatabase} size="sm" />
              </button>
              <button
                onClick={createNewChat}
                className="text-orange-600 hover:text-orange-700 transition"
                title="Tạo cuộc trò chuyện mới"
              >
                <FontAwesomeIcon icon={faPlus} size="sm" />
              </button>
            </div>
          </div>
          <ul className="menu text-xs space-y-1 overflow-y-auto flex-1 pr-2">
            {Object.entries(chats).length === 0 ? (
              <p className="text-sm text-gray-500 italic">Hiện chưa có cuộc hội thoại nào</p>
            ) : (
              Object.entries(chats)
                .reverse()
                .map(([chatId, chat]) => (
                  <li
                    key={chatId}
                    className={`rounded-lg p-2 transition relative group ${
                      currentChatId === chatId ? "chat-item-active" : "hover:bg-orange-100"
                    }`}
                  >
                    <button
                      onClick={() => switchChat(chatId)}
                      className="text-left text-gray-700 hover:text-gray-900 break-words w-full pr-6"
                    >
                      <FontAwesomeIcon icon={faMessage} className="mr-2" />
                      {chat.title}
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        deleteChat(chatId)
                      }}
                      className="absolute top-1 right-1 text-gray-400 hover:text-red-600 hover:bg-red-50 active:bg-red-100 transition rounded-full p-1 w-6 h-6 flex items-center justify-center"
                      title="Xóa cuộc trò chuyện"
                    >
                      <FontAwesomeIcon icon={faTrash} size="xs" />
                    </button>
                  </li>
                ))
            )}
          </ul>
        </div>
      </div>

      {/* Lightbox Modal */}
      {lightboxImage && (
        <div
          className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4 backdrop-blur-sm"
          onClick={() => setLightboxImage(null)}
        >
          <div className="relative max-w-7xl max-h-[90vh] w-full h-full flex items-center justify-center">
            <img
              src={lightboxImage}
              alt="Full size"
              className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            />
            <button
              onClick={() => setLightboxImage(null)}
              className="absolute top-4 right-4 bg-white/90 hover:bg-white text-gray-800 rounded-full w-10 h-10 flex items-center justify-center shadow-lg transition-all hover:scale-110"
              title="Đóng"
            >
              <FontAwesomeIcon icon={faTimes} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default ChatBot
