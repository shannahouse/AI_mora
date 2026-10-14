const GESTURES = {
  scissors: { name: "剪刀", emoji: "✌️", beats: "rock" },
  rock: { name: "石頭", emoji: "✊", beats: "paper" },
  paper: { name: "布", emoji: "✋", beats: "scissors" },
};
const STORAGE_KEY = "rps-wizard-samples-v1";
const MIN_SAMPLES = 8;
const FEATURE_SIZE = 32;
const CELL_SIZE = 8;
const ORIENTATION_BINS = 8;
const SAMPLE_LIMIT = 60;

const video = document.querySelector("#camera");
const canvas = document.querySelector("#capture-canvas");
const cameraButton = document.querySelector("#camera-button");
const cameraStatus = document.querySelector("#camera-status");
const cameraPlaceholder = document.querySelector("#camera-placeholder");
const trainingStatus = document.querySelector("#training-status");
const trainingPanel = document.querySelector("#training-panel");
const matchPanel = document.querySelector("#match-panel");
const playButton = document.querySelector("#play-button");
const readyBadge = document.querySelector("#ready-badge");
const result = document.querySelector("#result");
const resultLine = document.querySelector("#result-line");
const resultMessage = document.querySelector("#result-message");
const resetButton = document.querySelector("#reset-button");

let cameraStream;
let samples = loadSamples();

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
                sample.length === CELL_SIZE * CELL_SIZE * ORIENTATION_BINS &&
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
  readyBadge.textContent = trained ? "訓練完成，可以對戰" : "尚未完成訓練";
  readyBadge.classList.toggle("ready", trained);
  document.querySelector("#training-step").classList.toggle("active", !trained);
  document.querySelector("#match-step").classList.toggle("active", trained);
  playButton.disabled = !trained || !cameraStream;
  trainingPanel.classList.toggle("complete", trained);
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
      video: { facingMode: "user" },
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

  const side = Math.floor(Math.min(video.videoWidth, video.videoHeight) * 0.55);
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

  const feature = new Float32Array(CELL_SIZE * CELL_SIZE * ORIENTATION_BINS);
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
      const cellX = Math.floor(x / CELL_SIZE);
      const cellY = Math.floor(y / CELL_SIZE);
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
  const nearest = [];
  for (const [gesture, classSamples] of Object.entries(samples)) {
    for (const sample of classSamples) {
      let distance = 0;
      for (let index = 0; index < feature.length; index += 1) {
        const difference = feature[index] - sample[index];
        distance += difference * difference;
      }
      nearest.push({ gesture, distance });
    }
  }
  nearest.sort((first, second) => first.distance - second.distance);

  const votes = new Map(Object.keys(GESTURES).map((gesture) => [gesture, 0]));
  for (const neighbor of nearest.slice(0, 5)) {
    votes.set(neighbor.gesture, votes.get(neighbor.gesture) + 1);
  }
  return [...votes].sort((first, second) => second[1] - first[1])[0][0];
}

function playRound() {
  result.hidden = true;
  try {
    const playerGesture = predictGesture(makeFeatureVector());
    const computerGesture = GESTURES[playerGesture].beats;
    resultLine.textContent =
      `你出${GESTURES[playerGesture].name} ${GESTURES[playerGesture].emoji}　電腦出${GESTURES[computerGesture].name} ${GESTURES[computerGesture].emoji}`;
    resultMessage.textContent = "電腦獲勝！下回合再試一次！";
    result.hidden = false;
  } catch (error) {
    trainingStatus.textContent = error.message;
  }
}

cameraButton.addEventListener("click", () => {
  if (cameraStream) stopCamera();
  else void startCamera();
});

for (const button of document.querySelectorAll(".gesture-card")) {
  button.addEventListener("click", () => captureSample(button.dataset.gesture));
}

playButton.addEventListener("click", playRound);

resetButton.addEventListener("click", () => {
  if (!window.confirm("確定要清除所有手勢訓練資料嗎？")) return;
  samples = Object.fromEntries(Object.keys(GESTURES).map((gesture) => [gesture, []]));
  if (!saveSamples()) return;
  result.hidden = true;
  trainingStatus.textContent = "訓練資料已清除，請重新教我辨識三種手勢。";
  updateProgress();
});

window.addEventListener("pagehide", stopCamera);
updateProgress();
