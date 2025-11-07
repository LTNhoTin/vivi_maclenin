"use client"

import { useState, useRef, useEffect } from "react"
import ScaleLoader from "react-spinners/ScaleLoader"
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome"
import { faMessage, faTrash, faPlus } from "@fortawesome/free-solid-svg-icons"
import ReactMarkdown from "react-markdown"
import robot_img from "../assets/ic5.png"
import { sendMessageChatService } from "./chatbotService"
import LinkBox from "./LinkBox"
import commonQuestionsData from "../db/commonQuestions.json"

function ChatBot(props) {
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)
  const [timeOfRequest, setTimeOfRequest] = useState(0)
  const [promptInput, setPromptInput] = useState("")
  const [model, setModel] = useState("LegalBizAI_pro")
  const [isLoading, setIsLoad] = useState(false)
  const [isGen, setIsGen] = useState(false)
  const [counter, setCounter] = useState(0)

  const [chats, setChats] = useState({
    default: {
      id: "default",
      title: "Cuộc trò chuyện mới",
      createdAt: new Date(),
      messages: [
        [
          "start",
          [
            "Xin chào! Đây là ViVi, trợ lý đắc lực về luật doanh nghiệp của bạn! Bạn muốn tìm kiếm thông tin về điều gì? Đừng quên chọn mô hình phù hợp để mình có thể giúp bạn tìm kiếm thông tin chính xác nhất nha.",
            null,
            null,
          ],
        ],
      ],
    },
  })
  const [currentChatId, setCurrentChatId] = useState("default")

  const models = [
    {
      value: "LegalBizAI_pro",
      name: "LegalBizAI Pro",
    },
    {
      value: "LegalBizAI",
      name: "LegalBizAI",
    },
  ]

  const commonQuestions = commonQuestionsData

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
    if (promptInput !== "" && isLoading === false) {
      setTimeOfRequest(0)
      setIsGen(true)
      const userMessage = promptInput
      setPromptInput("")
      inputRef.current.style.height = "auto"
      setIsLoad(true)

      setChats((prev) => ({
        ...prev,
        [currentChatId]: {
          ...prev[currentChatId],
          messages: [...prev[currentChatId].messages, ["end", [userMessage, model]]],
        },
      }))

      try {
        const result = await sendMessageChatService(userMessage, model)

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
              "Xin chào! Đây là ViVi, trợ lý đắc lực về luật doanh nghiệp của bạn! Bạn muốn tìm kiếm thông tin về điều gì?",
              null,
              null,
            ],
          ],
        ],
      },
    }))
    setCurrentChatId(newChatId)
    setPromptInput("")
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

  const switchChat = (chatId) => {
    setCurrentChatId(chatId)
    setPromptInput("")
    inputRef.current.style.height = "auto"
  }

  const currentChat = chats[currentChatId]
  const dataChat = currentChat?.messages || []

  return (
    <div className="bg-gradient-to-r from-orange-50 to-orange-100 flex flex-col w-full h-screen">
      <div className="bg-gradient-to-r from-orange-100 to-orange-50 border-b-2 border-orange-200 py-4 px-6 shadow-sm">
        <h1 className="text-2xl md:text-3xl font-bold bg-[linear-gradient(90deg,hsl(var(--s))_0%,hsl(var(--sf))_9%,hsl(var(--pf))_42%,hsl(var(--p))_47%,hsl(var(--a))_100%)] bg-clip-text will-change-auto [-webkit-text-fill-color:transparent] [transform:translate3d(0,0,0)] motion-reduce:!tracking-normal">
          ViVi Chat
        </h1>
        <p className="text-sm text-gray-600 mt-1">Hỏi bất cứ điều gì về luật kinh doanh</p>
      </div>

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

      <div className="lg:hidden p-2 flex justify-center bg-gradient-to-r from-orange-50 to-orange-100">
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

      <div className="flex flex-1 gap-3 p-3 overflow-hidden">
        <div className="hidden lg:flex flex-col w-64 bg-gray-50 rounded-2xl p-3 shadow-md border border-gray-200 overflow-hidden">
          <h2 className="font-bold mb-3 text-sm bg-[linear-gradient(90deg,hsl(var(--s))_0%,hsl(var(--sf))_9%,hsl(var(--pf))_42%,hsl(var(--p))_47%,hsl(var(--a))_100%)] bg-clip-text will-change-auto [-webkit-text-fill-color:transparent]">
            Những câu hỏi phổ biến
          </h2>
          <ul className="menu text-xs space-y-2 overflow-y-auto flex-1 pr-2">
            {commonQuestions.map((mess, i) => (
              <li key={i} className="hover:bg-orange-100 rounded-lg p-2 cursor-pointer transition">
                <button
                  onClick={() => handleQuickQuestionClick(mess.question)}
                  className="text-left text-gray-700 hover:text-gray-900 break-words"
                >
                  <FontAwesomeIcon icon={faMessage} className="mr-2" />
                  {mess.question}
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
            rounded-3xl border-2 p-3 overflow-auto flex-1"
          >
            {dataChat.map((dataMessages, i) =>
              dataMessages[0] === "start" ? (
                <div className="chat chat-start drop-shadow-md" key={i}>
                  <div className="chat-image avatar">
                    <div className="w-8 lg:w-10 rounded-full border-2 border-blue-500">
                      <img className="scale-150" src={robot_img || "/placeholder.svg"} alt="avatar" />
                    </div>
                  </div>
                  <div className="chat-bubble chat-bubble-gradient-receive break-words">
                    <ReactMarkdown>{dataMessages[1][0]}</ReactMarkdown>
                    {dataMessages[1][1] && dataMessages[1][1].length > 0 && (
                      <>
                        <div className="divider m-0"></div>
                        <LinkBox links={dataMessages[1][1]} />
                      </>
                    )}
                  </div>
                </div>
              ) : (
                <div className="chat chat-end" key={i}>
                  <div className="chat-bubble shadow-xl chat-bubble-gradient-send">
                    {dataMessages[1][0]}

                    <>
                      <div className="divider m-0"></div>
                      {/* <p className="font-light text-xs text-cyan-50">Mô hình: {dataMessages[1][1]}</p> */}
                    </>
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

          <div className="grid bg-gradient-to-r from-orange-50 to-orange-100 p-2 rounded-lg gap-2 mt-2">
            <textarea
              placeholder="Nhập câu hỏi tại đây..."
              className="shadow-xl border-2 focus:outline-none px-2 rounded-2xl input-primary col-span-full md:col-span-11 textarea-auto-resize"
              onChange={onChangeHandler}
              onKeyDown={handleKeyDown}
              disabled={isGen}
              value={promptInput}
              ref={inputRef}
              rows="1"
              style={{ resize: "none", overflow: "hidden", lineHeight: "3" }}
            />
            <button
              disabled={isGen}
              onClick={sendMessageChat}
              className="drop-shadow-md rounded-2xl col-span-1 md:col-span-1 btn btn-active btn-primary btn-square btn-send"
            >
              <svg
                stroke="currentColor"
                fill="none"
                strokeWidth="2"
                viewBox="0 0 24 24"
                color="white"
                height="15px"
                width="15px"
                xmlns="http://www.w3.org/2000/svg"
              >
                <line x1="22" y1="2" x2="11" y2="13"></line>
                <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
              </svg>
            </button>
            <p className="text-xs col-span-full text-justify">
              <b>Lưu ý: </b>ViVi có thể mắc lỗi. Hãy kiểm tra các thông tin quan trọng!
            </p>
          </div>
        </div>

        <div className="hidden lg:flex flex-col w-64 bg-gray-50 rounded-2xl p-3 shadow-md border border-gray-200 overflow-hidden">
          <div className="flex items-center justify-between mb-3">
            <h2 className="font-bold text-sm bg-[linear-gradient(90deg,hsl(var(--s))_0%,hsl(var(--sf))_9%,hsl(var(--pf))_42%,hsl(var(--p))_47%,hsl(var(--a))_100%)] bg-clip-text will-change-auto [-webkit-text-fill-color:transparent]">
              Lịch sử trò chuyện
            </h2>
            <button
              onClick={createNewChat}
              className="text-orange-600 hover:text-orange-700 transition"
              title="Tạo cuộc trò chuyện mới"
            >
              <FontAwesomeIcon icon={faPlus} size="sm" />
            </button>
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
                    className={`rounded-lg p-2 transition flex items-center justify-between group ${
                      currentChatId === chatId ? "chat-item-active" : "hover:bg-orange-100"
                    }`}
                  >
                    <button
                      onClick={() => switchChat(chatId)}
                      className="text-left text-gray-700 hover:text-gray-900 break-words flex-1"
                    >
                      <FontAwesomeIcon icon={faMessage} className="mr-2" />
                      {chat.title}
                    </button>
                    <button
                      onClick={() => deleteChat(chatId)}
                      className="text-gray-400 hover:text-red-500 transition opacity-0 group-hover:opacity-100 ml-2"
                      title="Xóa cuộc trò chuyện"
                    >
                      <FontAwesomeIcon icon={faTrash} size="sm" />
                    </button>
                  </li>
                ))
            )}
          </ul>
        </div>
      </div>
    </div>
  )
}

export default ChatBot
