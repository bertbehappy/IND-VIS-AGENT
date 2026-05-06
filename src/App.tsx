import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Camera, AlertTriangle, CheckCircle, Activity, Thermometer, Droplets, Info, Play, Square, Upload, Clock, List, Video, VideoOff } from 'lucide-react';
import { GoogleGenAI, Type } from '@google/genai';
import { motion, AnimatePresence } from 'motion/react';

// Define the structure of our JSON logs matching the prompt
interface AgentLog {
  camera_id: string;
  timestamp: string;
  event_class: string;
  log_category: string;
  device_status: string;
  temperature_c?: number | null;
  humidity_percent?: number | null;
  screen_time?: string | null;
  confidence: number;
  alert_required: boolean;
  alert_level: 'info' | 'warning' | 'critical';
  alert_code?: string;
  alert_message?: string;
  needs_human_review: boolean;
  raw_text: string;
  anomaly_detected: boolean;
  anomaly_type?: string[];
  state_changed: boolean;
}

// schema setup for Gemini
const logSchema = {
    type: Type.OBJECT,
    properties: {
        event_class: { type: Type.STRING, description: "running, stopped, fault, abnormal_reading, vision_failure" },
        log_category: { type: Type.STRING, description: "normal_operation, state_transition, fault_event, sensor_read_failure, camera_abnormal, ocr_review_needed" },
        device_status: { type: Type.STRING },
        temperature_c: { type: Type.NUMBER },
        humidity_percent: { type: Type.NUMBER },
        screen_time: { type: Type.STRING },
        confidence: { type: Type.NUMBER },
        alert_required: { type: Type.BOOLEAN },
        alert_level: { type: Type.STRING, description: "info, warning, critical" },
        alert_code: { type: Type.STRING },
        alert_message: { type: Type.STRING },
        needs_human_review: { type: Type.BOOLEAN },
        raw_text: { type: Type.STRING },
        anomaly_detected: { type: Type.BOOLEAN },
        anomaly_type: { type: Type.ARRAY, items: { type: Type.STRING } },
        state_changed: { type: Type.BOOLEAN }
    },
    required: ["event_class", "log_category", "device_status", "confidence", "alert_required", "alert_level", "raw_text", "anomaly_detected"]
};

// System Instruction for Gemini
const getPrompt = (lastState?: any) => `
你是 24x7 工業設備視覺監控 AI Agent。
任務是監看設備面板畫面，辨識運轉狀態、讀取溫濕度、判定故障事件、產生告警與分類 Log。

【任務要求】
- 只根據影像內容判斷，不得臆測。
- 若畫面可判讀：
  1. 擷取狀態文字
  2. 擷取溫度與濕度
  3. 判斷是否為正常、停機、故障、數值異常、鏡頭異常
- 若畫面不可判讀：
  - 回傳 device_status = 無法辨識
  - anomaly_type 加入 ocr_uncertain 或 camera_view_abnormal

【狀態分類規則】
- 畫面顯示「運轉中」=> event_class = "running"
- 畫面顯示「停機」=> event_class = "stopped"
- 畫面顯示「故障」=> event_class = "fault"
- 若濕度顯示為破折號、空白或不完整 => event_class = "abnormal_reading"
- 若整個螢幕模糊、偏移、反光嚴重 => event_class = "vision_failure"

【告警邏輯】
- fault => alert_level = "critical" / alert_required = true
- stopped => alert_level = "warning" / alert_required = true
- abnormal_reading => alert_level = "warning" / alert_required = true

【Log 分類】
- normal_operation
- state_transition
- fault_event
- sensor_read_failure
- camera_abnormal
- ocr_review_needed

注意事項：
你不是一次性 OCR 工具，而是即時監控代理。
你需要比較目前影像與上一筆結果，檢查是否有狀態轉換、數值跳動、讀值中斷、畫面偏移。
若目前狀態與上一幀不同，請將 log_category 設為 "state_transition", state_changed 設為 true。
如果溫濕度或是任何數值沒有明確顯示數字，請忽略且不要猜測或自創數字。

以下是你「上一幀（或上一筆紀錄）」所產生的狀態：
${lastState ? JSON.stringify(lastState, null, 2) : "無上一筆資料，此為第一幀。"}

請以此作為基準來判斷本次畫面是否與之不同。
`;

// Helper for dynamic colors
const getAlertColors = (level?: string) => {
    switch (level) {
        case 'critical': return 'text-red-500 bg-red-500/10 border-red-500/50 shadow-[0_0_15px_rgba(239,68,68,0.3)]';
        case 'warning': return 'text-amber-500 bg-amber-500/10 border-amber-500/50 shadow-[0_0_15px_rgba(245,158,11,0.3)]';
        case 'info': return 'text-blue-500 bg-blue-500/10 border-blue-500/50 shadow-[0_0_15px_rgba(59,130,246,0.3)]';
        default: return 'text-emerald-500 bg-emerald-500/10 border-emerald-500/50 shadow-[0_0_15px_rgba(16,185,129,0.3)]';
    }
}

export default function App() {
  const [logs, setLogs] = useState<AgentLog[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [intervalSecs, setIntervalSecs] = useState(3);
  const [staticImage, setStaticImage] = useState<string | null>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const lastStateRef = useRef<AgentLog | null>(null);
  const isAnalyzing = useRef(false);

  // Initialize Gemini API
  // Vite injects process.env from define block in vite.config.ts
  const ai = new GoogleGenAI({ apiKey: (process.env as any).GEMINI_API_KEY });

  const startCamera = async () => {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        if (videoRef.current) {
            videoRef.current.srcObject = stream;
        }
        setIsCameraActive(true);
        setStaticImage(null);
    } catch (err) {
        console.error("Camera access denied", err);
        alert("無法存取攝影機。請確認權限或嘗試上傳測試影像。\nCamera access denied.");
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
        videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
  };

  const captureFrame = useCallback(() => {
    if (staticImage) return staticImage;
    if (!videoRef.current || !canvasRef.current || !isCameraActive) return null;
    
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    
    if (canvas.width === 0 || canvas.height === 0) return null;

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      return canvas.toDataURL('image/jpeg', 0.8);
    }
    return null;
  }, [staticImage, isCameraActive]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
         setStaticImage(event.target?.result as string);
         if(isCameraActive) stopCamera();
      };
      reader.readAsDataURL(file);
      // Reset input value so same file can be uploaded again if needed
      e.target.value = '';
    }
  };

  useEffect(() => {
    let timer: number | undefined;
    if (isMonitoring) {
       timer = window.setInterval(async () => {
           if (isAnalyzing.current) return;
           const base64Str = captureFrame();
           if (!base64Str) return;

           isAnalyzing.current = true;
           try {
               const base64Data = base64Str.split(',')[1];
               
               const response = await ai.models.generateContent({
                   model: 'gemini-2.5-flash',
                   contents: [
                       {
                           role: 'user',
                           parts: [
                               { inlineData: { data: base64Data, mimeType: 'image/jpeg' } },
                               { text: getPrompt(lastStateRef.current) }
                           ]
                       }
                   ],
                   config: {
                       responseMimeType: "application/json",
                       responseSchema: logSchema,
                       temperature: 0.1
                   }
               });

               const rawText = response.text;
               if (rawText) {
                   const parsed = JSON.parse(rawText) as AgentLog;
                   
                   const now = new Date();
                   const yyyy = now.getFullYear();
                   const mm = String(now.getMonth() + 1).padStart(2, '0');
                   const dd = String(now.getDate()).padStart(2, '0');
                   const HH = String(now.getHours()).padStart(2, '0');
                   const MM = String(now.getMinutes()).padStart(2, '0');
                   const SS = String(now.getSeconds()).padStart(2, '0');
                   
                   const logEntry: AgentLog = {
                       ...parsed,
                       camera_id: "CAM-01",
                       timestamp: `${yyyy}-${mm}-${dd} ${HH}:${MM}:${SS}`
                   };

                   lastStateRef.current = logEntry;
                   // Add at top 
                   setLogs(prev => [logEntry, ...prev].slice(0, 50));
               }
           } catch(err) {
               console.error("Frame Analysis Failed:", err);
           } finally {
               isAnalyzing.current = false;
           }
       }, intervalSecs * 1000);
    }
    return () => clearInterval(timer);
  }, [isMonitoring, intervalSecs, captureFrame, ai]);

  const toggleMonitoring = () => {
    if (!staticImage && !isCameraActive) {
      alert("請先開啟攝影機或上傳測試圖片！");
      return;
    }
    setIsMonitoring(!isMonitoring);
  };

  const latestLog = logs.length > 0 ? logs[0] : null;
  const isSteady = latestLog ? (latestLog.alert_level === 'info' || !latestLog.alert_level) && !latestLog.alert_required : true;

  return (
    <div className="min-h-screen bg-gray-950 font-mono text-gray-300 p-4 lg:p-8 selection:bg-cyan-500/30">
      
      {/* HEADER */}
      <header className="flex flex-col md:flex-row items-start md:items-center justify-between border-b border-gray-800 pb-4 mb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-3">
            <Camera className="w-7 h-7 text-cyan-400" />
            IND-VIS-AGENT <span className="text-gray-500 font-normal text-sm border border-gray-700 px-2 py-0.5 rounded ml-2">v1.2.0</span>
          </h1>
          <p className="text-sm text-gray-500 mt-1">24x7 視覺監控代理 | 支援溫濕度及狀態自動判讀</p>
        </div>
        <div className="flex items-center gap-4 mt-4 md:mt-0 bg-gray-900 border border-gray-800 px-4 py-2 rounded-lg">
          <div className="flex items-center gap-2">
            <span className="relative flex h-3 w-3">
              {isMonitoring && <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>}
              <span className={`relative inline-flex rounded-full h-3 w-3 ${isMonitoring ? 'bg-cyan-500' : 'bg-gray-600'}`}></span>
            </span>
            <span className="text-sm font-medium uppercase tracking-wider">{isMonitoring ? 'AGENT ACTIVE' : 'AGENT STANDBY'}</span>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* LEFT COLUMN: VISION INPUT */}
        <div className="lg:col-span-5 space-y-4">
          <div className="border border-gray-800 rounded-xl overflow-hidden bg-gray-900 relative">
            {/* Visual HUD Overlay */}
            <div className="absolute inset-0 z-10 pointer-events-none border border-cyan-500/20 m-4 rounded">
               {/* Corner Brackets */}
               <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-cyan-500/50"></div>
               <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-cyan-500/50"></div>
               <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-cyan-500/50"></div>
               <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-cyan-500/50"></div>
               
               {isMonitoring && (
                 <motion.div 
                   className="absolute left-0 top-1/2 w-full h-px bg-cyan-500/30"
                   animate={{ y: [-150, 150, -150] }}
                   transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                 />
               )}
            </div>

            {/* Video / Image element */}
            <div className="relative aspect-video bg-black flex items-center justify-center">
               {!staticImage && !isCameraActive && (
                  <div className="text-gray-600 flex flex-col items-center">
                    <VideoOff className="w-12 h-12 mb-2 opacity-50" />
                    <p className="text-sm">無影像來源</p>
                  </div>
               )}
               {staticImage && (
                  <img src={staticImage} alt="Static test" className="w-full h-full object-contain z-0" />
               )}
               <video 
                 ref={videoRef} 
                 autoPlay 
                 playsInline
                 muted
                 className={`w-full h-full object-cover z-0 ${(staticImage || !isCameraActive) ? 'hidden' : 'block'}`} 
               />
               <canvas ref={canvasRef} className="hidden" />
            </div>

            {/* Controls */}
            <div className="bg-gray-900 border-t border-gray-800 p-4 space-y-4">
               
               <div className="flex flex-wrap gap-3">
                 {!isCameraActive ? (
                   <button 
                     onClick={startCamera}
                     className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded text-sm flex items-center gap-2 transition-colors border border-gray-700 hover:border-gray-600"
                   >
                     <Video className="w-4 h-4" /> 開啟攝影機
                   </button>
                 ) : (
                   <button 
                     onClick={stopCamera}
                     className="px-4 py-2 bg-red-900/40 hover:bg-red-900/60 text-red-300 rounded text-sm flex items-center gap-2 transition-colors border border-red-900"
                   >
                     <Square className="w-4 h-4" /> 關閉攝影機
                   </button>
                 )}

                 <input 
                   type="file" 
                   accept="image/*" 
                   onChange={handleImageUpload} 
                   ref={fileInputRef}
                   className="hidden" 
                 />
                 <button 
                   onClick={() => fileInputRef.current?.click()}
                   className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded text-sm flex items-center gap-2 transition-colors border border-gray-700 hover:border-gray-600"
                 >
                   <Upload className="w-4 h-4" /> 上傳靜態圖
                 </button>
               </div>

               <div className="border-t border-gray-800 pt-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <label className="text-xs text-gray-500 uppercase tracking-widest flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      分析頻率: {intervalSecs}s
                    </label>
                    <input 
                      type="range" 
                      min="1" max="10" step="1" 
                      disabled={isMonitoring}
                      value={intervalSecs}
                      onChange={(e) => setIntervalSecs(parseInt(e.target.value))}
                      className="w-24 accent-cyan-500"
                    />
                  </div>
                  
                  <button
                    onClick={toggleMonitoring}
                    className={`px-6 py-2 rounded font-bold tracking-widest text-sm flex items-center gap-2 transition-all ${
                      isMonitoring 
                        ? 'bg-red-500 hover:bg-red-600 text-white shadow-[0_0_10px_rgba(239,68,68,0.5)]' 
                        : 'bg-cyan-500 hover:bg-cyan-400 text-gray-950 shadow-[0_0_10px_rgba(6,182,212,0.5)]'
                    }`}
                  >
                    {isMonitoring ? (
                      <><Square className="w-4 h-4 fill-current"/> 停止監控</>
                    ) : (
                      <><Play className="w-4 h-4 fill-current"/> 啟動監控</>
                    )}
                  </button>
               </div>
            </div>
          </div>

          {/* AI Metrics Hint */}
          <div className="p-4 bg-gray-900 border border-gray-800 rounded-xl text-xs text-gray-500 flex justify-between items-center">
            <span>Core: Gemini 2.5 Flash Vision</span>
            <span>Threshold: {intervalSecs}s Polling</span>
          </div>
        </div>


        {/* RIGHT COLUMN: STATE & LOGS */}
        <div className="lg:col-span-7 flex flex-col gap-6">
          
          {/* Main State Dashboard */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
            
            <div className="col-span-2 md:col-span-2 bg-gray-900 border border-gray-800 p-4 rounded-xl flex items-center justify-between">
               <div>
                  <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">設備狀態 (Status)</p>
                  <p className="text-2xl font-semibold text-white">
                    {latestLog ? latestLog.device_status || '未知' : '---'}
                  </p>
               </div>
               <Activity className={`w-8 h-8 opacity-50 ${isSteady ? 'text-emerald-500' : 'text-amber-500'}`} />
            </div>

            <div className="bg-gray-900 border border-gray-800 p-4 rounded-xl flex items-center justify-between">
               <div>
                  <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">系統溫度</p>
                  <p className="text-2xl font-semibold text-white">
                    {latestLog?.temperature_c != null ? `${latestLog.temperature_c}°C` : '---'}
                  </p>
               </div>
               <Thermometer className="w-6 h-6 text-gray-500" />
            </div>

            <div className="bg-gray-900 border border-gray-800 p-4 rounded-xl flex items-center justify-between">
               <div>
                  <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">環境濕度</p>
                  <p className="text-2xl font-semibold text-white">
                    {latestLog?.humidity_percent != null ? `${latestLog.humidity_percent}%` : '---'}
                  </p>
               </div>
               <Droplets className="w-6 h-6 text-gray-500" />
            </div>
            
          </div>

          {/* Alert Banner / Active Context */}
          <AnimatePresence mode="popLayout">
            {latestLog && latestLog.alert_required && (
              <motion.div 
                initial={{ opacity: 0, y: -20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className={`p-4 rounded-xl flex items-start gap-4 border ${getAlertColors(latestLog.alert_level)}`}
              >
                <AlertTriangle className="w-8 h-8 shrink-0 mt-1" />
                <div>
                  <h3 className="font-bold text-lg uppercase tracking-wide">
                    {latestLog.alert_level === 'critical' ? 'CRITICAL ALERT: ' : 'SYSTEM WARNING: '}
                    {latestLog.event_class}
                  </h3>
                  <p className="opacity-90 mt-1 text-sm">{latestLog.alert_message}</p>
                  {latestLog.anomaly_type && latestLog.anomaly_type.length > 0 && (
                    <div className="flex gap-2 mt-3 flex-wrap">
                      {latestLog.anomaly_type.map((t, i) => (
                        <span key={i} className="px-2 py-0.5 bg-black/30 rounded text-xs border border-current opacity-80 backdrop-blur">
                          {t}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Detailed Event Stream */}
          <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden flex-grow flex flex-col min-h-[400px]">
             <div className="p-3 border-b border-gray-800 bg-gray-900/50 flex items-center justify-between">
                <div className="flex items-center gap-2">
                   <List className="w-4 h-4 text-gray-500" />
                   <h3 className="text-sm font-semibold uppercase tracking-wider">即時分析日誌 (Live JSON Logs)</h3>
                </div>
                <div className="text-xs text-gray-500">
                  {logs.length} entries
                </div>
             </div>

             <div className="overflow-y-auto p-4 space-y-3 shrink-0 h-[400px]">
               {logs.length === 0 ? (
                 <div className="h-full flex flex-col items-center justify-center text-gray-600">
                   <Info className="w-8 h-8 mb-2 opacity-30" />
                   <p className="text-sm">等待分析結果...</p>
                 </div>
               ) : (
                 <AnimatePresence>
                   {logs.map((log, idx) => (
                      <motion.div 
                        key={`${log.timestamp}-${idx}`}
                        initial={idx === 0 ? { opacity: 0, x: -20 } : false}
                        animate={{ opacity: 1, x: 0 }}
                        className={`p-3 rounded border text-xs overflow-x-auto relative ${idx === 0 ? 'bg-gray-800 shadow-md border-gray-700' : 'bg-gray-950/50 border-gray-800/50 opacity-80'}`}
                      >
                         {idx === 0 && (
                           <div className="absolute top-2 right-2 flex items-center gap-1 text-[10px] text-cyan-500 border border-cyan-500/20 px-1.5 py-0.5 rounded bg-cyan-500/10">
                              <span className="w-1.5 h-1.5 bg-cyan-500 rounded-full animate-pulse"></span>
                              NEW
                           </div>
                         )}
                         <pre className={`${idx === 0 ? 'text-gray-200' : 'text-gray-400'}`}>
                           {JSON.stringify(log, null, 2)}
                         </pre>
                      </motion.div>
                   ))}
                 </AnimatePresence>
               )}
             </div>
          </div>

        </div>
      </div>
    </div>
  );
}
