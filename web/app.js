const GESTURES = {
  scissors: { name: "剪刀", emoji: "✌️", beats: "rock" },
  rock: { name: "石頭", emoji: "✊", beats: "paper" },
  paper: { name: "布", emoji: "✋", beats: "scissors" },
};
const STORAGE_KEY = "rps-wizard-samples-v4";
const MIN_SAMPLES = 12;
const SAMPLE_LIMIT = 100;
const PREDICTION_INTERVAL = 100;
const RESULT_DURATION = 4000;
const STABILITY_WINDOW = 5;
const STABLE_VOTES = 3;
const MEDIAPIPE_VERSION = "0.10.21";
const MEDIAPIPE_CDN = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const HAND_FEATURE_SIZE = (21 * 20) / 2;

const video = document.querySelector("#camera");
const cameraButton = document.querySelector("#camera-button");
const detectionButton = document.querySelector("#detection-button");
const playButton = document.querySelector("#play-button");
const cameraStatus = document.querySelector("#camera-status");
const cameraPlaceholder = document.querySelector("#camera-placeholder");
const detectionOverlay = document.querySelector("#detection-overlay");
const detectionContext = detectionOverlay.getContext("2d");
// MediaPipe 對非正方形畫面的關節座標會偏移，所以先裁出影片中央的正方形再送去偵測。
const detectionInput = document.createElement("canvas");
const detectionInputContext = detectionInput.getContext("2d");
const forceCpuDelegate = new URLSearchParams(location.search).has("cpu");
const calibrationButton = document.querySelector("#calibration-button");
const debugView = document.querySelector("#debug-view");
const debugViewContext = debugView.getContext("2d");
let activeDelegate = forceCpuDelegate ? "CPU" : "GPU";
const countdown = document.querySelector("#countdown");
const trainingStatus = document.querySelector("#training-status");
const trainingPanel = document.querySelector("#training-panel");
const result = document.querySelector("#result");
const resultEmoji = document.querySelector("#result-emoji");
const resultName = document.querySelector("#result-name");
const resetButton = document.querySelector("#reset-button");
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

let cameraStream;
let samples = loadSamples();
let matchPhase = "idle";
let matchRunId = 0;
let countdownUtterance;
let handLandmarker;
let handLandmarkerPromise;
let detectionOverlayEnabled = false;
let detectionAnimationFrame;
let latestHandDetection;
let latestDetectionAt = 0;
let lastDetectionVideoTime = -1;

function loadSamples() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    return Object.fromEntries(
      Object.keys(GESTURES).map((gesture) => [
        gesture,
        Array.isArray(saved[gesture])
          ? saved[gesture].filter(
              (sample) =>
                Array.isArray(sample) &&
                sample.length === HAND_FEATURE_SIZE &&
                sample.every(Number.isFinite),
            )
          : [],
      ]),
    );
  } catch (error) {
    console.error("無法讀取本機訓練資料。", error);
    trainingStatus.textContent = "無法讀取本機訓練資料，請清除資料後重新訓練。";
    return Object.fromEntries(Object.keys(GESTURES).map((gesture) => [gesture, []]));
  }
}

function saveSamples() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(samples));
  } catch (error) {
    console.error("無法儲存本機訓練資料。", error);
    trainingStatus.textContent = "瀏覽器無法儲存訓練資料，請檢查儲存空間設定。";
    return false;
  }
  return true;
}

function allGesturesTrained() {
  return Object.keys(GESTURES).every((gesture) => samples[gesture].length >= MIN_SAMPLES);
}

function updateProgress() {
  for (const gesture of Object.keys(GESTURES)) {
    const count = samples[gesture].length;
    document.querySelector(`#count-${gesture}`).textContent =
      `${count} / ${MIN_SAMPLES} 張`;
  }

  const trained = allGesturesTrained();
  document.querySelector("#training-step").classList.toggle("active", !trained);
  document.querySelector("#match-step").classList.toggle("active", trained);
  playButton.disabled = (!trained || !cameraStream) && matchPhase === "idle";
  trainingPanel.classList.toggle("complete", trained);
  if (matchPhase === "idle") {
    playButton.textContent = "開始對戰";
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    cameraStatus.textContent = "此瀏覽器無法存取攝影機，請使用最新版瀏覽器並從 localhost 開啟。";
    return;
  }

  cameraButton.disabled = true;
  cameraStatus.textContent = "正在等待攝影機權限…";
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    cameraStream = stream;
    video.srcObject = stream;
    await video.play();
    cameraStatus.textContent = "正在載入手部辨識模型（首次使用需要下載）…";
    await loadHandLandmarker();
    video.classList.add("active");
    cameraPlaceholder.hidden = true;
    cameraButton.textContent = "關閉攝影機";
    cameraButton.disabled = false;
    cameraStatus.textContent = "攝影機與手部辨識已就緒，請將一隻手放在框框中央。";
    trainingStatus.textContent = "辨識器已準備好；將一隻手放進框框，依序拍攝三種手勢樣本。";
    for (const button of document.querySelectorAll(".gesture-card")) {
      button.disabled = false;
    }
    updateProgress();
  } catch (error) {
    for (const track of stream?.getTracks() || []) track.stop();
    video.srcObject = null;
    cameraStream = undefined;
    cameraButton.disabled = false;
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      cameraStatus.textContent = "攝影機權限遭拒，請在瀏覽器網址列允許攝影機後重試。";
    } else if (error.name === "NotFoundError") {
      cameraStatus.textContent = "找不到攝影機，請連接網路攝影機後重試。";
    } else {
      console.error("手部辨識器或攝影機啟動失敗。", error);
      cameraStatus.textContent =
        `手部辨識器載入失敗，請確認網路連線後重試。${error.message ? ` (${error.message})` : ""}`;
    }
  }
  updateProgress();
}

async function loadHandLandmarker() {
  if (handLandmarker) return handLandmarker;
  if (!handLandmarkerPromise) {
    handLandmarkerPromise = (async () => {
      const { FilesetResolver, HandLandmarker } = await import(
        `${MEDIAPIPE_CDN}/vision_bundle.mjs`
      );
      const vision = await FilesetResolver.forVisionTasks(`${MEDIAPIPE_CDN}/wasm`);
      const options = {
        baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: forceCpuDelegate ? "CPU" : "GPU" },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.65,
        minHandPresenceConfidence: 0.65,
        minTrackingConfidence: 0.6,
      };
      try {
        handLandmarker = await HandLandmarker.createFromOptions(vision, options);
      } catch (gpuError) {
        console.warn("GPU 手部辨識初始化失敗，改用 CPU。", gpuError);
        activeDelegate = "CPU (GPU 失敗)";
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          ...options,
          baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: "CPU" },
        });
      }
      return handLandmarker;
    })();
  }
  try {
    return await handLandmarkerPromise;
  } catch (error) {
    handLandmarkerPromise = undefined;
    throw error;
  }
}

function stopCamera() {
  stopMatch();
  stopDetectionOverlay();
  if (cameraStream) {
    for (const track of cameraStream.getTracks()) track.stop();
    cameraStream = undefined;
  }
  video.srcObject = null;
  video.classList.remove("active");
  cameraPlaceholder.hidden = false;
  cameraButton.textContent = "開啟攝影機";
  cameraStatus.textContent = "攝影機已關閉。";
  trainingStatus.textContent = "開啟攝影機後即可繼續訓練。";
  for (const button of document.querySelectorAll(".gesture-card")) {
    button.disabled = true;
  }
  updateProgress();
}

function makeFeatureVector() {
  if (!cameraStream || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    throw new Error("攝影機畫面尚未就緒，請稍候再試。");
  }

  const detection =
    detectionOverlayEnabled &&
    latestHandDetection &&
    performance.now() - latestDetectionAt < 200
    ? latestHandDetection
    : detectHandFrame();
  const landmarks = detection.worldLandmarks[0];
  if (!landmarks) return undefined;

  const palmScale = Math.hypot(
    landmarks[9].x - landmarks[0].x,
    landmarks[9].y - landmarks[0].y,
    landmarks[9].z - landmarks[0].z,
  );
  if (!Number.isFinite(palmScale) || palmScale < 0.001) return undefined;

  const feature = [];
  for (let first = 0; first < landmarks.length; first += 1) {
    for (let second = first + 1; second < landmarks.length; second += 1) {
      const dx = landmarks[first].x - landmarks[second].x;
      const dy = landmarks[first].y - landmarks[second].y;
      const dz = landmarks[first].z - landmarks[second].z;
      feature.push(Math.hypot(dx, dy, dz) / palmScale);
    }
  }
  return feature;
}

function captureDetectionInput() {
  const size = Math.min(video.videoWidth, video.videoHeight);
  const x = (video.videoWidth - size) / 2;
  const y = (video.videoHeight - size) / 2;
  if (detectionInput.width !== size) {
    detectionInput.width = size;
    detectionInput.height = size;
  }
  detectionInputContext.drawImage(video, x, y, size, size, 0, 0, size, size);
  return { x, y, size };
}

function detectHandFrame() {
  const crop = captureDetectionInput();
  latestHandDetection = handLandmarker.detectForVideo(detectionInput, performance.now());
  latestHandDetection.crop = crop;
  latestDetectionAt = performance.now();
  lastDetectionVideoTime = video.currentTime;
  if (detectionOverlayEnabled) drawDetectionOverlay(latestHandDetection);
  if (!debugView.hidden) drawDebugView(latestHandDetection);
  return latestHandDetection;
}

function drawDetectionOverlay(detection) {
  const bounds = detectionOverlay.getBoundingClientRect();
  const pixelRatio = window.devicePixelRatio || 1;
  const canvasWidth = Math.round(bounds.width * pixelRatio);
  const canvasHeight = Math.round(bounds.height * pixelRatio);
  if (detectionOverlay.width !== canvasWidth || detectionOverlay.height !== canvasHeight) {
    detectionOverlay.width = canvasWidth;
    detectionOverlay.height = canvasHeight;
  }

  const context = detectionContext;
  context.clearRect(0, 0, canvasWidth, canvasHeight);
  const landmarks = detection.landmarks[0];
  if (!landmarks || !video.videoWidth || !video.videoHeight) return;

  // 以影片元素實際在畫面上的位置換算（object-fit: cover 置中裁切），不假設它和畫布一樣大。
  // 影片和畫布都用 scaleX(-1) 鏡像，所以水平方向以右緣對齊。
  const videoBounds = video.getBoundingClientRect();
  const boxWidth = videoBounds.width * pixelRatio;
  const boxHeight = videoBounds.height * pixelRatio;
  const scale = Math.max(boxWidth / video.videoWidth, boxHeight / video.videoHeight);
  const offsetX =
    (bounds.right - videoBounds.right) * pixelRatio + (boxWidth - video.videoWidth * scale) / 2;
  const offsetY =
    (videoBounds.top - bounds.top) * pixelRatio + (boxHeight - video.videoHeight * scale) / 2;
  const { crop } = detection;
  const point = (index) => ({
    x: offsetX + (crop.x + landmarks[index].x * crop.size) * scale,
    y: offsetY + (crop.y + landmarks[index].y * crop.size) * scale,
  });

  drawSkeleton(context, landmarks.length, point, pixelRatio);
}

function drawSkeleton(context, jointCount, point, pixelRatio) {
  context.lineWidth = 2.5 * pixelRatio;
  context.strokeStyle = "#43f5c4";
  context.lineCap = "round";
  context.lineJoin = "round";
  context.beginPath();
  for (const [start, end] of HAND_CONNECTIONS) {
    const first = point(start);
    const second = point(end);
    context.moveTo(first.x, first.y);
    context.lineTo(second.x, second.y);
  }
  context.stroke();

  context.fillStyle = "#ff1744";
  for (let index = 0; index < jointCount; index += 1) {
    const joint = point(index);
    context.beginPath();
    context.arc(joint.x, joint.y, 4 * pixelRatio, 0, Math.PI * 2);
    context.fill();
  }
}

// 手勢定位校準：在畫面角落顯示實際送進 AI 的正方形畫面，骨架直接畫在上面，不經過任何畫面換算。
function drawDebugView(detection) {
  const size = 240;
  const lineHeight = 15;
  const info = [
    `delegate: ${activeDelegate}`,
    `video: ${video.videoWidth}×${video.videoHeight}`,
    `element: ${video.clientWidth}×${video.clientHeight}  dpr: ${window.devicePixelRatio}`,
    `fit: ${getComputedStyle(video).objectFit} / ${getComputedStyle(video).objectPosition}`,
  ];
  if (debugView.width !== size) {
    debugView.width = size;
    debugView.height = size + info.length * lineHeight + 8;
  }
  const context = debugViewContext;
  context.fillStyle = "#000";
  context.fillRect(0, 0, debugView.width, debugView.height);
  context.drawImage(detectionInput, 0, 0, size, size);
  const landmarks = detection.landmarks[0];
  if (landmarks) {
    const point = (index) => ({ x: landmarks[index].x * size, y: landmarks[index].y * size });
    drawSkeleton(context, landmarks.length, point, 0.6);
  }
  context.fillStyle = "#fff";
  context.font = "11px monospace";
  info.forEach((line, index) => context.fillText(line, 6, size + 4 + (index + 1) * lineHeight - 3));
}

function updateDetectionOverlay() {
  if (!detectionOverlayEnabled) return;
  if (!cameraStream) {
    stopDetectionOverlay();
    return;
  }

  if (
    video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
    video.currentTime !== lastDetectionVideoTime &&
    performance.now() - latestDetectionAt >= PREDICTION_INTERVAL
  ) {
    try {
      detectHandFrame();
    } catch (error) {
      console.error("AI 手部關節顯示失敗。", error);
      stopDetectionOverlay();
      cameraStatus.textContent = `AI 偵測顯示失敗：${error.message}`;
      return;
    }
  }
  detectionAnimationFrame = window.requestAnimationFrame(updateDetectionOverlay);
}

function startDetectionOverlay() {
  detectionOverlayEnabled = true;
  detectionOverlay.hidden = false;
  detectionOverlay.classList.add("active");
  detectionButton.textContent = "關閉 AI 偵測";
  detectionButton.setAttribute("aria-pressed", "true");
  lastDetectionVideoTime = -1;
  updateDetectionOverlay();
}

function stopDetectionOverlay() {
  detectionOverlayEnabled = false;
  if (detectionAnimationFrame !== undefined) {
    window.cancelAnimationFrame(detectionAnimationFrame);
    detectionAnimationFrame = undefined;
  }
  detectionOverlay.hidden = true;
  detectionOverlay.classList.remove("active");
  detectionButton.textContent = "顯示 AI 偵測";
  detectionButton.setAttribute("aria-pressed", "false");
  setCalibrationView(false);
  latestHandDetection = undefined;
  latestDetectionAt = 0;
  lastDetectionVideoTime = -1;
  detectionContext.clearRect(0, 0, detectionOverlay.width, detectionOverlay.height);
}

function captureSample(gesture) {
  try {
    const feature = makeFeatureVector();
    if (!feature) {
      trainingStatus.textContent = "尚未找到手部，請將一隻手完整放進攝影機畫面的框框內。";
      return;
    }
    const classSamples = samples[gesture];
    if (classSamples.length >= SAMPLE_LIMIT) classSamples.shift();
    classSamples.push(feature);
    if (!saveSamples()) return;
    updateProgress();
    trainingStatus.textContent = `${GESTURES[gesture].name}手勢樣本已儲存（${classSamples.length} 張）；可稍微改變手掌角度後再拍攝。`;
  } catch (error) {
    trainingStatus.textContent = error.message;
  }
}

function predictGesture(feature) {
  const classDistances = [];
  for (const [gesture, classSamples] of Object.entries(samples)) {
    const closest = [];
    for (const sample of classSamples) {
      let distance = 0;
      for (let index = 0; index < feature.length; index += 1) {
        const difference = feature[index] - sample[index];
        distance += difference * difference;
      }
      closest.push(distance);
      closest.sort((first, second) => first - second);
      if (closest.length > 3) closest.pop();
    }
    if (closest.length === 0) continue;
    const averageDistance =
      closest.reduce((sum, distance) => sum + distance, 0) / closest.length;
    classDistances.push({ gesture, distance: averageDistance });
  }
  classDistances.sort((first, second) => first.distance - second.distance);
  const [best, runnerUp] = classDistances;
  const margin =
    (runnerUp.distance - best.distance) / Math.max(runnerUp.distance, 0.001);
  if (margin < 0.04) return undefined;
  return best.gesture;
}

function showResult(playerGesture) {
  const computerGesture = GESTURES[playerGesture].beats;
  resultEmoji.textContent = GESTURES[computerGesture].emoji;
  resultName.textContent = GESTURES[computerGesture].name;
  result.hidden = false;
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

async function startMatch() {
  if (matchPhase !== "idle" || !cameraStream || !allGesturesTrained()) return;

  matchPhase = "countdown";
  const runId = ++matchRunId;
  playButton.disabled = false;
  playButton.textContent = "停止對戰";
  void runMatchLoop(runId);
}

async function runCountdown(runId) {
  result.hidden = true;
  for (const number of [3, 2, 1]) {
    if (runId !== matchRunId) return;
    countdown.textContent = number;
    countdown.hidden = false;
    await delay(1000);
  }
  countdown.hidden = true;
  return runId === matchRunId;
}

function speakGestureCue(runId) {
  if (!window.speechSynthesis || typeof SpeechSynthesisUtterance !== "function") {
    cameraStatus.textContent = "此瀏覽器不支援語音播放，請使用最新版 Chrome。";
    return Promise.resolve(false);
  }

  const utterance = new SpeechSynthesisUtterance("剪刀、石頭、布");
  utterance.lang = "zh-TW";
  utterance.rate = 1.2;
  utterance.volume = 1;
  countdownUtterance = utterance;
  cameraStatus.textContent = "剪刀、石頭、布！";

  return new Promise((resolve) => {
    const finish = (played) => {
      if (countdownUtterance === utterance) countdownUtterance = undefined;
      resolve(played);
    };
    utterance.onend = () => finish(true);
    utterance.onerror = () => {
      if (runId === matchRunId) {
        cameraStatus.textContent = "無法播放猜拳語音，請確認瀏覽器音量與語音設定後重試。";
      }
      finish(false);
    };

    try {
      window.speechSynthesis.speak(utterance);
    } catch (error) {
      if (runId === matchRunId) {
        cameraStatus.textContent = `語音播放失敗：${error.message}`;
      }
      finish(false);
    }
  });
}

async function runMatchLoop(runId) {
  while (runId === matchRunId) {
    matchPhase = "countdown";
    if (!(await runCountdown(runId))) return;
    if (!(await speakGestureCue(runId))) {
      if (runId === matchRunId) {
        matchPhase = "idle";
        updateProgress();
      }
      return;
    }
    if (runId !== matchRunId) return;

    matchPhase = "running";
    cameraStatus.textContent = "辨識中，請將手勢放在框框中央並保持清楚。";
    const gesture = await detectGesture(runId);
    if (!gesture || runId !== matchRunId) return;

    showResult(gesture);
    matchPhase = "result";
    cameraStatus.textContent = `已辨識你的手勢：${GESTURES[gesture].name}。4 秒後開始下一局。`;
    await delay(RESULT_DURATION);
  }
}

async function detectGesture(runId) {
  const predictions = [];

  while (runId === matchRunId && matchPhase === "running") {
    try {
      const feature = makeFeatureVector();
      const gesture = feature ? predictGesture(feature) : undefined;
      predictions.push(gesture);
      if (predictions.length > STABILITY_WINDOW) predictions.shift();

      const counts = new Map();
      for (const prediction of predictions) {
        if (prediction) counts.set(prediction, (counts.get(prediction) || 0) + 1);
      }
      const [stableGesture, votes] = [...counts].sort(
        (first, second) => second[1] - first[1],
      )[0] || [];

      if (votes >= STABLE_VOTES) return stableGesture;
      if (!feature) {
        cameraStatus.textContent = "尚未找到手部，請將一隻手完整放進框框內。";
      } else {
        cameraStatus.textContent = "辨識中，請將手勢放在框框中央並保持清楚。";
      }
    } catch (error) {
      cameraStatus.textContent = error.message;
    }
    await delay(PREDICTION_INTERVAL);
  }
  return undefined;
}

function stopMatch() {
  matchRunId += 1;
  if (countdownUtterance) {
    window.speechSynthesis.cancel();
    countdownUtterance = undefined;
  }
  const wasRunning = matchPhase !== "idle";
  matchPhase = "idle";
  countdown.hidden = true;
  if (wasRunning) result.hidden = true;
  if (wasRunning) cameraStatus.textContent = "對戰已停止。";
  updateProgress();
}

cameraButton.addEventListener("click", () => {
  if (cameraStream) stopCamera();
  else void startCamera();
});

detectionButton.addEventListener("click", () => {
  if (detectionOverlayEnabled) {
    stopDetectionOverlay();
  } else if (!cameraStream || !handLandmarker) {
    cameraStatus.textContent = "請先開啟攝影機並等待 AI 辨識模型準備完成。";
  } else {
    startDetectionOverlay();
  }
});

// 校準畫面靠 AI 偵測的迴圈更新，所以打開時一併開啟 AI 偵測，關閉 AI 偵測時也一併關閉。
function setCalibrationView(visible) {
  debugView.hidden = !visible;
  calibrationButton.setAttribute("aria-pressed", String(visible));
}

calibrationButton.addEventListener("click", () => {
  if (!debugView.hidden) {
    setCalibrationView(false);
  } else if (!cameraStream || !handLandmarker) {
    cameraStatus.textContent = "請先開啟攝影機並等待 AI 辨識模型準備完成。";
  } else {
    if (!detectionOverlayEnabled) startDetectionOverlay();
    setCalibrationView(true);
  }
});

for (const button of document.querySelectorAll(".gesture-card")) {
  button.addEventListener("click", () => captureSample(button.dataset.gesture));
}

playButton.addEventListener("click", () => {
  if (matchPhase === "idle") void startMatch();
  else stopMatch();
});

resetButton.addEventListener("click", () => {
  if (!window.confirm("確定要清除所有手勢訓練資料嗎？")) return;
  stopMatch();
  samples = Object.fromEntries(Object.keys(GESTURES).map((gesture) => [gesture, []]));
  if (!saveSamples()) return;
  result.hidden = true;
  trainingStatus.textContent = "訓練資料已清除，請重新教我辨識三種手勢。";
  updateProgress();
});

window.addEventListener("pagehide", stopCamera);
updateProgress();
