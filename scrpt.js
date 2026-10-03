// Face tracker: MediaPipe Face Landmarker running in the browser.
// The library and model are downloaded once; your video stays on your device.

const VISION_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const stage = $("stage");
const placeholder = $("placeholder");
const message = $("message");
const toggleBtn = $("toggleBtn");

let lib = null;          // { FaceLandmarker, FilesetResolver, DrawingUtils }
let landmarker = null;
let drawing = null;
let stream = null;
let running = false;
let lastVideoTime = -1;
let loadedFaces = 0;     // numFaces the current landmarker was built with
let frames = 0;
let fpsStart = performance.now();
let blinkCount = 0;
let eyesClosed = false;
let swapping = false;   // true while the model is being rebuilt

const opt = {
  mesh: $("optMesh"),
  outline: $("optOutline"),
  box: $("optBox"),
  mirror: $("optMirror"),
  faces: $("optFaces"),
};

function showMessage(text) {
  message.textContent = text;
  placeholder.classList.remove("hidden");
}

async function createLandmarker() {
  const faces = Number(opt.faces.value);
  if (landmarker && loadedFaces === faces) return;

  if (!lib) {
    // Dynamic import so this works as a normal script (no ES-module CORS issues).
    lib = await import(`${VISION_URL}/vision_bundle.mjs`);
  }
  const fileset = await lib.FilesetResolver.forVisionTasks(`${VISION_URL}/wasm`);
  if (landmarker) landmarker.close();
  landmarker = await lib.FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numFaces: faces,
    outputFaceBlendshapes: true,
  });
  loadedFaces = faces;
  drawing = new lib.DrawingUtils(ctx);
}

async function start() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showMessage("Camera access needs https:// or http://localhost. Open this page from a local server.");
    return;
  }
  toggleBtn.disabled = true;
  showMessage("Loading face model...");
  try {
    await createLandmarker();
    showMessage("Waiting for camera permission...");
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    placeholder.classList.add("hidden");
    running = true;
    blinkCount = 0;
    toggleBtn.textContent = "Stop camera";
    toggleBtn.classList.add("stop");
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    const denied = err && err.name === "NotAllowedError";
    showMessage(
      denied
        ? "Camera permission was blocked. Allow camera access for this page and try again."
        : "Couldn't start: " + (err && err.message ? err.message : err)
    );
  } finally {
    toggleBtn.disabled = false;
  }
}

function stop() {
  running = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  toggleBtn.textContent = "Start camera";
  toggleBtn.classList.remove("stop");
  $("faceCount").textContent = "0 faces";
  $("fps").textContent = "0 fps";
  showMessage("Camera is off");
}

function loop() {
  if (!running) return;

  if (!swapping && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    render(result);

    frames++;
    const now = performance.now();
    if (now - fpsStart >= 1000) {
      $("fps").textContent = `${Math.round((frames * 1000) / (now - fpsStart))} fps`;
      frames = 0;
      fpsStart = now;
    }
  }
  requestAnimationFrame(loop);
}

function render(result) {
  const faces = result.faceLandmarks || [];
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $("faceCount").textContent = `${faces.length} ${faces.length === 1 ? "face" : "faces"}`;

  const { FaceLandmarker } = lib;
  faces.forEach((pts) => {
    if (opt.mesh.checked) {
      drawing.drawConnectors(pts, FaceLandmarker.FACE_LANDMARKS_TESSELATION, {
        color: "rgba(90, 242, 192, 0.28)",
        lineWidth: 0.6,
      });
    }
    if (opt.outline.checked) {
      const style = { color: "#5af2c0", lineWidth: 2 };
      [
        FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
        FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
        FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
        FaceLandmarker.FACE_LANDMARKS_LIPS,
        FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
      ].forEach((set) => drawing.drawConnectors(pts, set, style));
    }
    if (opt.box.checked) drawBox(pts);
  });

  if (faces.length) {
    const shapes = result.faceBlendshapes && result.faceBlendshapes[0];
    updateReadouts(faces[0], shapes ? shapes.categories : []);
  } else {
    resetReadouts();
  }
}

function drawBox(pts) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of pts) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  const pad = 0.02;
  const x = (minX - pad) * canvas.width;
  const y = (minY - pad) * canvas.height;
  const w = (maxX - minX + pad * 2) * canvas.width;
  const h = (maxY - minY + pad * 2) * canvas.height;
  const c = Math.min(w, h) * 0.18; // corner length

  ctx.strokeStyle = "#5af2c0";
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(x, y + c); ctx.lineTo(x, y); ctx.lineTo(x + c, y);
  ctx.moveTo(x + w - c, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + c);
  ctx.moveTo(x + w, y + h - c); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - c, y + h);
  ctx.moveTo(x + c, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - c);
  ctx.stroke();
}

// ---- Readouts ---------------------------------------------------------

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function score(categories, name) {
  const c = categories.find((k) => k.categoryName === name);
  return c ? c.score : 0;
}

function setFill(id, fraction) {
  $(id).style.left = "0";
  $(id).style.width = `${clamp(fraction, 0, 1) * 100}%`;
}

function setCentered(id, value) {
  // value in -1..1; bar grows left or right from the middle
  const v = clamp(value, -1, 1);
  const el = $(id);
  el.style.left = `${50 + Math.min(0, v) * 50}%`;
  el.style.width = `${Math.abs(v) * 50}%`;
}

function updateReadouts(pts, cats) {
  const smile = (score(cats, "mouthSmileLeft") + score(cats, "mouthSmileRight")) / 2;
  const mouth = score(cats, "jawOpen");
  setFill("mSmile", smile);
  setFill("mMouth", mouth);
  $("vSmile").textContent = `${Math.round(smile * 100)}%`;
  $("vMouth").textContent = `${Math.round(mouth * 100)}%`;

  // Blink counter: count each closed -> open transition
  const closed = (score(cats, "eyeBlinkLeft") + score(cats, "eyeBlinkRight")) / 2 > 0.5;
  if (eyesClosed && !closed) blinkCount++;
  eyesClosed = closed;
  $("vBlink").textContent = blinkCount;

  // Head turn: where the nose sits between the two sides of the face.
  // Landmarks: 1 = nose tip, 234 / 454 = face edges, 33 / 263 = outer eye corners.
  const nose = pts[1], a = pts[234], b = pts[454];
  const lo = Math.min(a.x, b.x), hi = Math.max(a.x, b.x);
  let ratio = (nose.x - lo) / (hi - lo || 1);   // 0..1 in camera space
  if (opt.mirror.checked) ratio = 1 - ratio;    // match what's on screen
  const turn = (ratio - 0.5) * 2.4;             // about -1..1 across the useful range
  setCentered("mTurn", turn);
  $("vTurn").textContent = turn < -0.12 ? "left" : turn > 0.12 ? "right" : "center";

  // Head tilt: angle of the line through the eyes, positive = clockwise on screen.
  const e1 = pts[33], e2 = pts[263];
  let tilt = (Math.atan2((e2.y - e1.y) * canvas.height, (e2.x - e1.x) * canvas.width) * 180) / Math.PI;
  if (opt.mirror.checked) tilt = -tilt;
  setCentered("mTilt", tilt / 45);
  $("vTilt").textContent = `${Math.round(tilt)}\u00B0`;
}

function resetReadouts() {
  ["mSmile", "mMouth", "mTurn", "mTilt"].forEach((id) => {
    $(id).style.left = "0";
    $(id).style.width = "0";
  });
  $("vSmile").textContent = "0%";
  $("vMouth").textContent = "0%";
  $("vTurn").textContent = "-";
  $("vTilt").textContent = "-";
}

// ---- Wiring -----------------------------------------------------------

toggleBtn.addEventListener("click", () => (running ? stop() : start()));

opt.mirror.addEventListener("change", () => stage.classList.toggle("mirror", opt.mirror.checked));
stage.classList.toggle("mirror", opt.mirror.checked);

opt.faces.addEventListener("change", async () => {
  if (!running) return;
  swapping = true;
  try {
    await createLandmarker();
  } catch (err) {
    console.error(err);
  } finally {
    swapping = false;
  }
});

window.addEventListener("pagehide", () => {
  if (stream) stream.getTracks().forEach((t) => t.stop());
});
