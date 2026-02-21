/**
 * Face Morph Studio — Frontend Logic
 * Handles webcam, API calls, and UI state management.
 */

// ==================== DOM Elements ====================
const webcam = document.getElementById('webcam');
const captureCanvas = document.getElementById('captureCanvas');
const videoOverlay = document.getElementById('videoOverlay');
const fileUpload = document.getElementById('fileUpload');

const btnStartCamera = document.getElementById('btnStartCamera');
const btnCapture = document.getElementById('btnCapture');
const btnApplyMorph = document.getElementById('btnApplyMorph');
const btnSurpriseMe = document.getElementById('btnSurpriseMe');
const btnRandomAgain = document.getElementById('btnRandomAgain');
const btnDownload = document.getElementById('btnDownload');
const btnTryAnother = document.getElementById('btnTryAnother');
const btnReset = document.getElementById('btnReset');

const randomQuote = document.getElementById('randomQuote');
const criminalBgToggle = document.getElementById('criminalBgToggle');

const captureSection = document.getElementById('captureSection');
const detectionSection = document.getElementById('detectionSection');
const effectsSection = document.getElementById('effectsSection');
const resultSection = document.getElementById('resultSection');

const originalImage = document.getElementById('originalImage');
const detectedImage = document.getElementById('detectedImage');
const detectionInfo = document.getElementById('detectionInfo');
const resultOriginal = document.getElementById('resultOriginal');
const resultMorphed = document.getElementById('resultMorphed');

const effectsGrid = document.getElementById('effectsGrid');
const strengthSlider = document.getElementById('strengthSlider');
const strengthValue = document.getElementById('strengthValue');

const loadingOverlay = document.getElementById('loadingOverlay');
const loadingText = document.getElementById('loadingText');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toastMessage');

// ==================== State ====================
let stream = null;
let capturedImageData = null;  // base64 data URL of captured/uploaded image
let selectedEffect = 'bulge';
let morphedImageData = null;

// ==================== Camera ====================
btnStartCamera.addEventListener('click', async () => {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: 640, height: 480 }
        });
        webcam.srcObject = stream;
        videoOverlay.classList.add('hidden');
        btnCapture.disabled = false;
        btnStartCamera.innerHTML = '<span class="btn-icon">⏹</span> Stop Camera';
        btnStartCamera.onclick = stopCamera;
    } catch (err) {
        showToast('Camera access denied. Please allow camera permissions.');
        console.error('Camera error:', err);
    }
});

function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }
    webcam.srcObject = null;
    videoOverlay.classList.remove('hidden');
    btnCapture.disabled = true;
    btnStartCamera.innerHTML = '<span class="btn-icon">📹</span> Start Camera';
    btnStartCamera.onclick = null;
    btnStartCamera.addEventListener('click', async () => {
        location.reload(); // Simple reload to re-bind
    });
}

// ==================== Capture ====================
btnCapture.addEventListener('click', () => {
    if (!stream) return;

    captureCanvas.width = webcam.videoWidth;
    captureCanvas.height = webcam.videoHeight;
    const ctx = captureCanvas.getContext('2d');
    ctx.drawImage(webcam, 0, 0);

    capturedImageData = captureCanvas.toDataURL('image/jpeg', 0.92);
    stopCamera();
    detectFace();
});

// ==================== File Upload ====================
fileUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (ev) => {
        capturedImageData = ev.target.result;
        detectFace();
    };
    reader.readAsDataURL(file);
});

// ==================== Face Detection ====================
async function detectFace() {
    if (!capturedImageData) return;

    showLoading('Detecting faces with YOLO...');

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: capturedImageData }),
        });

        const data = await response.json();
        hideLoading();

        if (data.error && !data.annotated_image) {
            showToast(data.error);
            return;
        }

        // Show detection section
        originalImage.src = capturedImageData;
        detectedImage.src = data.annotated_image;
        detectionInfo.innerHTML = `✅ Detected <strong>${data.faces_count}</strong> face(s) — Landmarks: ${data.has_landmarks ? '✓ Ready' : '✗ Not available'}`;

        detectionSection.classList.remove('hidden');
        detectionSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        // Show effects section
        buildEffectsGrid();
        effectsSection.classList.remove('hidden');

    } catch (err) {
        hideLoading();
        showToast('Error connecting to server. Make sure Flask is running.');
        console.error('Detection error:', err);
    }
}

// ==================== Effects Grid ====================
function buildEffectsGrid() {
    const effects = [
        { id: 'bulge', emoji: '👁', name: 'Bulge', desc: 'Fish-eye inflation' },
        { id: 'cartoon', emoji: '🎨', name: 'Cartoon', desc: 'Comic book style' },
        { id: 'squeeze', emoji: '🤏', name: 'Squeeze', desc: 'Tall & thin' },
        { id: 'big_eyes', emoji: '👀', name: 'Big Eyes', desc: 'Enlarged eyes' },
        { id: 'wide_smile', emoji: '😁', name: 'Wide Smile', desc: 'Exaggerated grin' },
    ];

    effectsGrid.innerHTML = '';
    effects.forEach((eff, idx) => {
        const card = document.createElement('div');
        card.className = 'effect-card' + (idx === 0 ? ' selected' : '');
        card.dataset.effect = eff.id;
        card.innerHTML = `
            <span class="effect-emoji">${eff.emoji}</span>
            <div class="effect-name">${eff.name}</div>
            <div class="effect-desc">${eff.desc}</div>
        `;
        card.addEventListener('click', () => selectEffect(eff.id));
        effectsGrid.appendChild(card);
    });
}

function selectEffect(effectId) {
    selectedEffect = effectId;
    document.querySelectorAll('.effect-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.effect === effectId);
    });
}

// ==================== Strength Slider ====================
strengthSlider.addEventListener('input', () => {
    strengthValue.textContent = strengthSlider.value;
});

// ==================== Apply Morph ====================
btnApplyMorph.addEventListener('click', () => applyMorphEffect(selectedEffect));

// ==================== Surprise Me! ====================
const RANDOM_QUOTES = [
    "🎲 Rolled the dice... your face is doomed!",
    "🌀 Going full chaos mode!",
    "💫 The universe chose this for you.",
    "😂 Brace yourself...",
    "🎭 Random vibes only!",
    "🔮 The morph gods have spoken.",
    "🤪 This one's on the algorithm!",
    "🎰 Jackpot! Your face won the lottery.",
    "🤖 AI says: trust the process.",
    "✨ Fate picked this. Don't blame me.",
];

function pickRandomEffect() {
    const effects = ['bulge', 'cartoon', 'squeeze', 'big_eyes', 'wide_smile'];
    // Prefer a different effect than the current one
    const others = effects.filter(e => e !== selectedEffect);
    const pick = others[Math.floor(Math.random() * others.length)];
    return pick;
}

function doSurprise() {
    const randomEffect = pickRandomEffect();
    // Pick random strength between 0.8 and 1.8
    const randomStrength = (Math.random() * 1.0 + 0.8).toFixed(1);
    strengthSlider.value = randomStrength;
    strengthValue.textContent = randomStrength;

    // Visually highlight the picked effect card
    selectEffect(randomEffect);

    // Show a fun quote
    const quote = RANDOM_QUOTES[Math.floor(Math.random() * RANDOM_QUOTES.length)];
    randomQuote.textContent = quote;
    randomQuote.classList.remove('hidden');

    // Auto apply after a brief moment so user sees the card highlight
    setTimeout(() => applyMorphEffect(randomEffect), 300);
}

btnSurpriseMe.addEventListener('click', doSurprise);
btnRandomAgain.addEventListener('click', doSurprise);

// ==================== Apply Morph (shared function) ====================
async function applyMorphEffect(effect) {
    if (!capturedImageData) return;

    showLoading(`Applying ${effect} morph...`);

    try {
        const response = await fetch('/morph', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: capturedImageData,
                effect: effect,
                strength: parseFloat(strengthSlider.value),
                criminal_bg: criminalBgToggle.checked,
            }),
        });

        const data = await response.json();
        hideLoading();

        if (data.error) {
            showToast(data.error);
            return;
        }

        morphedImageData = data.morphed_image;

        // Show result
        resultOriginal.src = capturedImageData;
        resultMorphed.src = morphedImageData;
        resultSection.classList.remove('hidden');
        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

        showToast('Morph applied successfully! 🎉');

    } catch (err) {
        hideLoading();
        showToast('Error applying morph effect.');
        console.error('Morph error:', err);
    }
}

// ==================== Download ====================
btnDownload.addEventListener('click', () => {
    if (!morphedImageData) return;

    const link = document.createElement('a');
    link.href = morphedImageData;
    link.download = `morph_${selectedEffect}_${Date.now()}.jpg`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    // Also save on server
    fetch('/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image: morphedImageData,
            effect: selectedEffect,
        }),
    }).then(res => res.json()).then(data => {
        if (data.success) {
            showToast(`Saved: ${data.filename}`);
        }
    }).catch(() => { });
});

// ==================== Try Another Effect ====================
btnTryAnother.addEventListener('click', () => {
    resultSection.classList.add('hidden');
    effectsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
});

// ==================== Reset ====================
btnReset.addEventListener('click', () => {
    capturedImageData = null;
    morphedImageData = null;
    selectedEffect = 'bulge';

    detectionSection.classList.add('hidden');
    effectsSection.classList.add('hidden');
    resultSection.classList.add('hidden');

    videoOverlay.classList.remove('hidden');
    btnCapture.disabled = true;

    captureSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    location.reload();
});

// ==================== Loading ====================
function showLoading(text = 'Processing...') {
    loadingText.textContent = text;
    loadingOverlay.classList.remove('hidden');
}

function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

// ==================== Toast ====================
function showToast(message, duration = 3500) {
    toastMessage.textContent = message;
    toast.classList.remove('hidden');
    clearTimeout(window._toastTimer);
    window._toastTimer = setTimeout(() => {
        toast.classList.add('hidden');
    }, duration);
}
