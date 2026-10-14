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
const playButton = document.querySelector("#play-button");
const cameraStatus = document.querySelector("#camera-status");
const cameraPlaceholder = document.querySelector("#camera-placeholder");
const countdown = document.querySelector("#countdown");
const trainingStatus = document.querySelector("#training-status");
const trainingPanel = document.querySelector("#training-panel");
const result = document.querySelector("#result");
const resultEmoji = document.querySelector("#result-emoji");
const resultName = document.querySelector("#result-name");
const resetButton = document.querySelector("#reset-button");

let cameraStream;
let samples = loadSamples();
let matchPhase = "idle";
let matchRunId = 0;
let handLandmarker;
let handLandmarkerPromise;

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
        baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: "GPU" },
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

  const landmarks = handLandmarker.detectForVideo(video, performance.now()).worldLandmarks[0];
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

async function runMatchLoop(runId) {
  while (runId === matchRunId) {
    matchPhase = "countdown";
    if (!(await runCountdown(runId))) return;

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
