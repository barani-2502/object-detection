import { useEffect, useRef, useState, useCallback } from 'react';
import { ObjectDetector, FilesetResolver } from '@mediapipe/tasks-vision';
import type { ObjectDetectorResult } from '@mediapipe/tasks-vision';
import './App.css';

interface LogEntry {
  time: string;
  msg: string;
  type: 'info' | 'success' | 'warn' | 'error';
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const detectorRef = useRef<ObjectDetector | null>(null);
  const [device, setDevice] = useState<'GPU' | 'CPU'>('GPU');
  const [model, setModel] = useState<'efficientdet_lite0' | 'ssd_mobilenet_v2'>('efficientdet_lite0');
  const [precision, setPrecision] = useState<'float32' | 'float16' | 'int8'>('float16');
  const [detections, setDetections] = useState<any[]>([]);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [testMode, setTestMode] = useState<'webcam' | 'image'>('webcam');
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const requestRef = useRef<number | null>(null);

  // --- NEW PERFORMANCE STATES ---
  const [fps, setFps] = useState<number>(0);
  const [latency, setLatency] = useState<number>(0);
  const lastFrameTime = useRef<number>(performance.now());

  const addLog = useCallback((msg: string, type: LogEntry['type'] = 'info') => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: true });
    setLogs(prev => [{ time, msg, type }, ...prev].slice(0, 20));
  }, []);

  useEffect(() => {
    let instance: ObjectDetector | null = null;
    let isActive = true;

    const initDetector = async () => {
      setIsLoading(true);
      addLog(`Initializing MediaPipe Graph with ${device}...`, "info");
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
        );

          instance = await ObjectDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/object_detector/${model}/${precision}/1/${model}.tflite`,
            delegate: device
          },
          scoreThreshold: 0.3,
          runningMode: testMode === 'webcam' ? "VIDEO" : "IMAGE"
        });

        if (isActive) {
          detectorRef.current = instance;
          setIsLoading(false);
          addLog(`Graph successfully started running on ${device}.`, "success");
        }
      } catch (err) {
        if (isActive) {
          addLog(`Failed to initialize TFLite delegate on ${device}.`, "error");
          console.error(err);
          setIsLoading(false);
        }
      }
    };

    initDetector();
    return () => {
      isActive = false;
      detectorRef.current = null;
      if (instance) {
        instance.close();
      }
    };
  }, [addLog, device, precision, model, testMode]);

  // Clean up animation frame purely on unmount
  useEffect(() => {
    return () => {
      if (requestRef.current) {
        cancelAnimationFrame(requestRef.current);
        requestRef.current = null;
      }
    };
  }, []);

  const runImageInference = useCallback(async () => {
    if (detectorRef.current && imageRef.current) {
      const startTimeMs = performance.now();
      try {
        const result: ObjectDetectorResult = detectorRef.current.detect(imageRef.current);
        const endTimeMs = performance.now();
        const currentLatency = endTimeMs - startTimeMs;

        setLatency(currentLatency);
        // FPS is not relevant for static images, but we could set it to 0 or N/A
        setFps(0);

        const targets = ['cell phone', 'book', 'laptop', 'person'];
        const filtered = result.detections.filter((d: any) =>
          targets.includes(d.categories[0].categoryName || '')
        );

        setDetections(filtered);
        addLog(`Image inference complete: ${filtered.length} objects found.`, "success");
      } catch (e) {
        addLog("Error during image inference.", "error");
        console.error(e);
      }
    }
  }, [addLog]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setSelectedImage(url);
      setDetections([]);
      addLog(`Image uploaded: ${file.name}`, "info");

      // We need to wait for the image to load before running inference
      // This will be handled by an onLoad handler on the image element
    }
  };

  const predictWebcam = useCallback(() => {
    if (detectorRef.current && videoRef.current && videoRef.current.readyState >= 2) {
      // --- PERFORMANCE METRIC START ---
      const startTimeMs = performance.now();

      try {
        // Inference
        const result: ObjectDetectorResult = detectorRef.current.detectForVideo(videoRef.current, startTimeMs);

        const endTimeMs = performance.now();
        const currentLatency = endTimeMs - startTimeMs;

      // FPS Calculation
      const frameDelta = endTimeMs - lastFrameTime.current;
      const currentFps = 1000 / frameDelta;

      setLatency(currentLatency);
      setFps(currentFps);
      lastFrameTime.current = endTimeMs;
      // ---------------------------------

        const targets = ['cell phone', 'book', 'laptop', 'person'];
        const filtered = result.detections.filter((d: any) =>
          targets.includes(d.categories[0].categoryName || '')
        );

        setDetections(filtered);
      } catch (e) {
        // Ignore inference errors (e.g. from hot-swapping closed detector)
      }
    }
    requestRef.current = requestAnimationFrame(predictWebcam);
  }, []);

  const enableCam = async () => {
    if (!detectorRef.current && !isLoading) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          setIsCameraActive(true);
          addLog("Webcam stream active.", "success");
          if (!requestRef.current) {
            predictWebcam();
          }
        };
      }
    } catch (err) {
      addLog("Webcam access denied.", "error");
    }
  };

  return (
    <div className="dashboard-container">
      <aside className="side-panel">
        <div className="panel-header">System Configuration</div>
        <div className="config-content">
          <div className="config-item"><span>Engine</span> <span>MediaPipe</span></div>
          <div className="config-item">
            <span>Model</span>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value as 'efficientdet_lite0' | 'ssd_mobilenet_v2')}
              className="device-select"
            >
              <option value="efficientdet_lite0">EfficientDet Lite0</option>
              <option value="ssd_mobilenet_v2">SSD MobileNet V2</option>
            </select>
          </div>
          <div className="config-item">
            <span>Delegate</span>
            <select
              value={device}
              onChange={(e) => setDevice(e.target.value as 'GPU' | 'CPU')}
              className="device-select"
            >
              <option value="GPU">GPU</option>
              <option value="CPU">CPU</option>
            </select>
          </div>
          <div className="config-item">
            <span>Test Mode</span>
            <select
              value={testMode}
              onChange={(e) => {
                setTestMode(e.target.value as 'webcam' | 'image');
                setDetections([]);
              }}
              className="device-select"
            >
              <option value="webcam">Webcam Feed</option>
              <option value="image">Image Upload</option>
            </select>
          </div>
          <div className="config-item">
            <span>Precision</span>
            <select
              value={precision}
              onChange={(e) => setPrecision(e.target.value as 'float32' | 'float16' | 'int8')}
              className="device-select"
            >
              <option value="float32">Float 32</option>
              <option value="float16">Float 16</option>
              <option value="int8">Int 8</option>
            </select>
          </div>

          {/* --- PERFORMANCE DISPLAY --- */}
          <div className="config-item">
            <span>Inference Latency</span>
            <span className={latency > 50 ? "warn" : "green"}>{latency.toFixed(1)} ms</span>
          </div>
          <div className="config-item">
            <span>Throughput</span>
            <span className={fps < 20 ? "warn" : "green"}>{fps.toFixed(1)} FPS</span>
          </div>
          {/* -------------------------- */}

          <div className="config-item"><span>Resolution</span> <span>640x480</span></div>
          <div className="config-item"><span>JS Heap</span> <span>{(performance as any).memory?.usedJSHeapSize ? `${(Math.round((performance as any).memory.usedJSHeapSize / 1048576))} MB` : 'N/A'}</span></div>
        </div>
      </aside>

      <main className={`monitor-panel ${isLoading ? 'invisible' : ''}`}>
        <div className="panel-header">{testMode === 'webcam' ? 'Webcam Monitor' : 'Image Monitor'}</div>
        <div className="monitor-container">
          {testMode === 'webcam' ? (
            <>
              {!isCameraActive && (
                <button className="mdc-button" onClick={enableCam} disabled={isLoading}>
                  {isLoading ? "LOADING..." : "ENABLE WEBCAM"}
                </button>
              )}

              <div className="video-viewport">
                <video ref={videoRef} autoPlay playsInline muted />
                {detections.map((det, index) => {
                  if (!videoRef.current) return null;
                  const { width, height, originX, originY } = det.boundingBox;
                  const displayWidth = videoRef.current.offsetWidth;
                  const displayHeight = videoRef.current.offsetHeight;
                  const naturalWidth = videoRef.current.videoWidth || 1;
                  const naturalHeight = videoRef.current.videoHeight || 1;

                  const sX = displayWidth / naturalWidth;
                  const sY = displayHeight / naturalHeight;

                  const boxWidth = width * sX;
                  const boxHeight = height * sY;
                  const boxX = displayWidth - boxWidth - (originX * sX);
                  const boxY = originY * sY;

                  return (
                    <div key={index} className="detection-box" style={{ left: boxX, top: boxY, width: boxWidth, height: boxHeight }}>
                      <span className="confidence-tag">
                        {det.categories[0].categoryName} - {Math.round(det.categories[0].score * 100)}%
                      </span>
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <>
              <input
                type="file"
                accept="image/*"
                ref={fileInputRef}
                style={{ display: 'none' }}
                onChange={handleImageUpload}
              />
              {!selectedImage && (
                <button className="mdc-button" onClick={() => fileInputRef.current?.click()} disabled={isLoading}>
                  {isLoading ? "LOADING..." : "UPLOAD TEST IMAGE"}
                </button>
              )}

              {selectedImage && (
                <div className="image-test-wrapper">
                  <div className="video-viewport static-viewport">
                    <img
                      ref={imageRef}
                      src={selectedImage}
                      alt="Upload"
                      onLoad={runImageInference}
                      style={{ width: '100%', display: 'block' }}
                    />
                    {detections.map((det, index) => {
                      if (!imageRef.current) return null;
                      const { width, height, originX, originY } = det.boundingBox;
                      const displayWidth = imageRef.current.offsetWidth;
                      const displayHeight = imageRef.current.offsetHeight;
                      const naturalWidth = imageRef.current.naturalWidth || 1;
                      const naturalHeight = imageRef.current.naturalHeight || 1;

                      const sX = displayWidth / naturalWidth;
                      const sY = displayHeight / naturalHeight;

                      const boxWidth = width * sX;
                      const boxHeight = height * sY;
                      const boxX = originX * sX;
                      const boxY = originY * sY;

                      return (
                        <div key={index} className="detection-box" style={{ left: boxX, top: boxY, width: boxWidth, height: boxHeight }}>
                          <span className="confidence-tag">
                            {det.categories[0].categoryName} - {Math.round(det.categories[0].score * 100)}%
                          </span>
                        </div>
                      );
                    })}
                  </div>
                  <button className="mdc-button secondary-btn" onClick={() => fileInputRef.current?.click()}>
                    CHANGE IMAGE
                  </button>
                </div>
              )}
            </>
          )}
        </div>
        <div className="status-bar">
          {detections.some(d => d.categories[0].categoryName !== 'person') ? (
            <h2 className="alert">VIOLATION DETECTED</h2>
          ) : (
            <h2 className="secure">ENVIRONMENT SECURE</h2>
          )}
        </div>
      </main>

      <aside className="side-panel console-panel">
        <div className="panel-header console-header">
          <span>Console Output</span>
          <button className="clear-btn" onClick={() => setLogs([])}>Clear</button>
        </div>
        <div className="console-body">
          {logs.map((log, i) => (
            <div key={i} className={`log-line ${log.type}`}>
              <span className="log-time">{log.time}</span>
              <span className="log-msg">{log.msg}</span>
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}

export default App;