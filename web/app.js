const GESTURES = {
  scissors: { name: "剪刀", emoji: "✌️", beats: "rock" },
  rock: { name: "石頭", emoji: "✊", beats: "paper" },
  paper: { name: "布", emoji: "✋", beats: "scissors" },
};
const STORAGE_KEY = "rps-wizard-samples-v2";
const MIN_SAMPLES = 12;
const FEATURE_SIZE = 48;
const CELL_SIZE = 6;
const ORIENTATION_BINS = 9;
const SAMPLE_LIMIT = 100;
const PREDICTION_INTERVAL = 100;
const RESULT_DURATION = 4000;
const STABILITY_WINDOW = 5;
const STABLE_VOTES = 3;

const video = document.querySelector("#camera");
const canvas = document.querySelector("#capture-canvas");
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
                sample.length ===
                  (FEATURE_SIZE / CELL_SIZE) ** 2 * ORIENTATION_BINS &&
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
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
    video.srcObject = cameraStream;
    await video.play();
    video.classList.add("active");
    cameraPlaceholder.hidden = true;
    cameraButton.textContent = "關閉攝影機";
    cameraButton.disabled = false;
    cameraStatus.textContent = "攝影機已就緒，請將手勢放在框框中央。";
    trainingStatus.textContent = "比出剪刀、石頭或布，再按下對應按鈕拍攝樣本。";
    for (const button of document.querySelectorAll(".gesture-card")) {
      button.disabled = false;
    }
  } catch (error) {
    cameraStream = undefined;
    cameraButton.disabled = false;
    if (error.name === "NotAllowedError" || error.name === "SecurityError") {
      cameraStatus.textContent = "攝影機權限遭拒，請在瀏覽器網址列允許攝影機後重試。";
    } else if (error.name === "NotFoundError") {
      cameraStatus.textContent = "找不到攝影機，請連接網路攝影機後重試。";
    } else {
      console.error("攝影機啟動失敗。", error);
      cameraStatus.textContent = `攝影機啟動失敗：${error.message || "請確認裝置正常後重試。"}`;
    }
  }
  updateProgress();
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

  const side = Math.floor(Math.min(video.videoWidth, video.videoHeight) * 0.75);
  const left = Math.floor((video.videoWidth - side) / 2);
  const top = Math.floor((video.videoHeight - side) / 2);
  canvas.width = FEATURE_SIZE;
  canvas.height = FEATURE_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(video, left, top, side, side, 0, 0, FEATURE_SIZE, FEATURE_SIZE);

  const { data } = context.getImageData(0, 0, FEATURE_SIZE, FEATURE_SIZE);
  const luminance = new Float32Array(FEATURE_SIZE * FEATURE_SIZE);
  for (let index = 0; index < luminance.length; index += 1) {
    const pixel = index * 4;
    luminance[index] =
      (0.299 * data[pixel] + 0.587 * data[pixel + 1] + 0.114 * data[pixel + 2]) / 255;
  }

  const cellsPerSide = FEATURE_SIZE / CELL_SIZE;
  const feature = new Float32Array(cellsPerSide * cellsPerSide * ORIENTATION_BINS);
  for (let y = 1; y < FEATURE_SIZE - 1; y += 1) {
    for (let x = 1; x < FEATURE_SIZE - 1; x += 1) {
      const offset = y * FEATURE_SIZE + x;
      const dx = luminance[offset + 1] - luminance[offset - 1];
      const dy = luminance[offset + FEATURE_SIZE] - luminance[offset - FEATURE_SIZE];
      const magnitude = Math.hypot(dx, dy);
      if (magnitude === 0) continue;

      let angle = Math.atan2(dy, dx);
      if (angle < 0) angle += Math.PI;
      if (angle >= Math.PI) angle -= Math.PI;
      const bin = Math.min(
        ORIENTATION_BINS - 1,
        Math.floor((angle / Math.PI) * ORIENTATION_BINS),
      );
      const cellX = Math.min(cellsPerSide - 1, Math.floor(x / CELL_SIZE));
      const cellY = Math.min(cellsPerSide - 1, Math.floor(y / CELL_SIZE));
      feature[(cellY * CELL_SIZE + cellX) * ORIENTATION_BINS + bin] += magnitude;
    }
  }

  let norm = 0;
  for (const value of feature) norm += value * value;
  norm = Math.sqrt(norm);
  if (norm < 0.001) throw new Error("畫面太暗或沒有清楚的手勢，請調整光線後重試。");
  return Array.from(feature, (value) => value / norm);
}

function captureSample(gesture) {
  try {
    const feature = makeFeatureVector();
    const classSamples = samples[gesture];
    if (classSamples.length >= SAMPLE_LIMIT) classSamples.shift();
    classSamples.push(feature);
    if (!saveSamples()) return;
    updateProgress();
    trainingStatus.textContent = `${GESTURES[gesture].name}樣本已儲存（${classSamples.length} 張）；可變換手勢角度後再拍攝。`;
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
      const gesture = predictGesture(makeFeatureVector());
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
