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
  const [detector, setDetector] = useState<ObjectDetector | null>(null);
  const [detections, setDetections] = useState<any[]>([]);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const requestRef = useRef<number | null>(null);

  // Helper to add logs to the console panel
  const addLog = useCallback((msg: string, type: LogEntry['type'] = 'info') => {
    const time = new Date().toLocaleTimeString('en-US', { hour12: true });
    setLogs(prev => [{ time, msg, type }, ...prev].slice(0, 20));
  }, []);

  // Initialize MediaPipe Detector
  useEffect(() => {
    const initDetector = async () => {
      addLog("Initializing MediaPipe Graph...", "info");
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
        );

        const instance = await ObjectDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite`,
            delegate: "GPU"
          },
          scoreThreshold: 0.2, // Slightly lower to catch more objects
          runningMode: "VIDEO"
        });

        setDetector(instance);
        setIsLoading(false);
        addLog("Graph successfully started running.", "success");
      } catch (err) {
        addLog("Failed to initialize TFLite delegate.", "error");
        console.error(err);
      }
    };

    initDetector();
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [addLog]);

  const predictWebcam = useCallback(() => {
    if (detector && videoRef.current && videoRef.current.readyState >= 2) {
      const startTimeMs = performance.now();
      const result: ObjectDetectorResult = detector.detectForVideo(videoRef.current, startTimeMs);

      const targets = ['cell phone', 'book', 'laptop', 'person'];
      const filtered = result.detections.filter((d: any) =>
        targets.includes(d.categories[0].categoryName || '')
      );

      filtered.forEach((d: any) => {
        const label = d.categories[0].categoryName;
        if (label !== 'person' && detections.length === 0) {
          addLog(`Violation: ${label.toUpperCase()} detected!`, "warn");
        }
      });

      setDetections(filtered);
    }
    requestRef.current = requestAnimationFrame(predictWebcam);
  }, [detector, detections.length, addLog]);

  const enableCam = async () => {
    if (!detector) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          setIsCameraActive(true);
          addLog("Webcam stream active.", "success");
          predictWebcam();
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
          <div className="config-item"><span>WebGL Support</span> <span className="green">Yes</span></div>
          <div className="config-item"><span>GPU Delegate</span> <span className="green">Enabled</span></div>
          <div className="config-item"><span>Resolution</span> <span>640x480</span></div>
          <div className="config-item"><span>Active Device</span> <span className="small">HP HD Camera</span></div>
          <div className="config-item"><span>JS Heap</span> <span>12.4 MB</span></div>
        </div>
      </aside>

      <main className={`monitor-panel ${isLoading ? 'invisible' : ''}`}>
        <div className="panel-header">Webcam Monitor</div>
        <div className="monitor-container">
          {!isCameraActive && (
            <button className="mdc-button" onClick={enableCam} disabled={isLoading}>
              {isLoading ? "LOADING..." : "ENABLE WEBCAM"}
            </button>
          )}

          <div className="video-viewport">
            <video ref={videoRef} autoPlay playsInline muted />
            {detections.map((det, index) => {
              const { width, height, originX, originY } = det.boundingBox;
              const displayX = videoRef.current ? videoRef.current.offsetWidth - width - originX : originX;
              const rawLabel = det.categories[0].categoryName;

              let displayLabel = rawLabel;

              return (
                <div key={index} className="detection-box" style={{ left: displayX, top: originY, width, height }}>
                  <span className="confidence-tag">
                    {displayLabel} - {Math.round(det.categories[0].score * 100)}%
                  </span>
                </div>
              );
            })}
          </div>
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