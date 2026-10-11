/**
 * MNIST Neural Vision - Canvas Engine & Prediction Handler
 */

document.addEventListener('DOMContentLoaded', () => {
    // -------------------------------------------------------------
    // DOM Elements
    // -------------------------------------------------------------
    const canvas = document.getElementById('digitCanvas');
    const ctx = canvas.getContext('2d');
    const canvasHint = document.getElementById('canvasHint');
    const canvasWrapper = document.querySelector('.canvas-wrapper');

    const penToolBtn = document.getElementById('penToolBtn');
    const eraserToolBtn = document.getElementById('eraserToolBtn');
    const brushSizeSlider = document.getElementById('brushSize');
    const brushSizeVal = document.getElementById('brushSizeVal');
    const undoBtn = document.getElementById('undoBtn');
    const clearBtn = document.getElementById('clearBtn');
    const predictBtn = document.getElementById('predictBtn');
    const autoPredictCheck = document.getElementById('autoPredictCheck');

    const samplesContainer = document.getElementById('samplesContainer');
    const predictedDigitDisplay = document.getElementById('predictedDigitDisplay');
    const confidenceVal = document.getElementById('confidenceVal');
    const confidencePill = document.getElementById('confidencePill');
    const processedImgPreview = document.getElementById('processedImgPreview');
    const latencyVal = document.getElementById('latencyVal');
    const probBarsList = document.getElementById('probBarsList');
    const topCandidatesList = document.getElementById('topCandidatesList');

    // -------------------------------------------------------------
    // Canvas Setup & High-DPI Scaling
    // -------------------------------------------------------------
    const CANVAS_SIZE = 280;
    const DPR = window.devicePixelRatio || 1;

    canvas.width = CANVAS_SIZE * DPR;
    canvas.height = CANVAS_SIZE * DPR;
    ctx.scale(DPR, DPR);

    // Initial canvas fill: Black background
    function initCanvas() {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
    }
    initCanvas();

    // -------------------------------------------------------------
    // Drawing State
    // -------------------------------------------------------------
    let isDrawing = false;
    let lastX = 0;
    let lastY = 0;
    let currentTool = 'pen'; // 'pen' or 'eraser'
    let brushSize = parseInt(brushSizeSlider.value, 10);
    let undoStack = [];
    const MAX_UNDO = 20;

    function saveUndoState() {
        if (undoStack.length >= MAX_UNDO) {
            undoStack.shift();
        }
        undoStack.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    }

    function checkCanvasHasDrawing() {
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        for (let i = 0; i < imgData.length; i += 4) {
            // Check red/green/blue channels for non-zero (white strokes)
            if (imgData[i] > 30 || imgData[i + 1] > 30 || imgData[i + 2] > 30) {
                return true;
            }
        }
        return false;
    }

    // -------------------------------------------------------------
    // Mouse / Touch Event Handlers
    // -------------------------------------------------------------
    function getCanvasCoordinates(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: clientX - rect.left,
            y: clientY - rect.top
        };
    }

    function startDrawing(e) {
        e.preventDefault();
        isDrawing = true;
        saveUndoState();
        canvasWrapper.classList.add('active');
        canvasHint.classList.add('hidden');

        const { x, y } = getCanvasCoordinates(e);
        lastX = x;
        lastY = y;

        // Draw initial point
        ctx.beginPath();
        ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
        ctx.fillStyle = currentTool === 'eraser' ? '#000000' : '#ffffff';
        ctx.fill();
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();

        const { x, y } = getCanvasCoordinates(e);

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        // Smooth curve
        const midX = (lastX + x) / 2;
        const midY = (lastY + y) / 2;
        ctx.quadraticCurveTo(lastX, lastY, midX, midY);
        ctx.lineTo(x, y);

        ctx.strokeStyle = currentTool === 'eraser' ? '#000000' : '#ffffff';
        ctx.lineWidth = brushSize;
        ctx.stroke();

        lastX = x;
        lastY = y;
    }

    let predictDebounceTimer = null;

    function stopDrawing() {
        if (!isDrawing) return;
        isDrawing = false;
        canvasWrapper.classList.remove('active');

        if (autoPredictCheck.checked) {
            clearTimeout(predictDebounceTimer);
            predictDebounceTimer = setTimeout(predictCurrentDigit, 120);
        }
    }

    canvas.addEventListener('mousedown', startDrawing);
    window.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDrawing);

    canvas.addEventListener('touchstart', startDrawing, { passive: false });
    window.addEventListener('touchmove', draw, { passive: false });
    window.addEventListener('touchend', stopDrawing);

    // -------------------------------------------------------------
    // Toolbar Controls
    // -------------------------------------------------------------
    penToolBtn.addEventListener('click', () => {
        currentTool = 'pen';
        penToolBtn.classList.add('active');
        eraserToolBtn.classList.remove('active');
    });

    eraserToolBtn.addEventListener('click', () => {
        currentTool = 'eraser';
        eraserToolBtn.classList.add('active');
        penToolBtn.classList.remove('active');
    });

    brushSizeSlider.addEventListener('input', (e) => {
        brushSize = parseInt(e.target.value, 10);
        brushSizeVal.textContent = brushSize;
    });

    undoBtn.addEventListener('click', () => {
        if (undoStack.length > 0) {
            const prevState = undoStack.pop();
            ctx.putImageData(prevState, 0, 0);

            if (!checkCanvasHasDrawing()) {
                canvasHint.classList.remove('hidden');
                resetResultsDisplay();
            } else if (autoPredictCheck.checked) {
                predictCurrentDigit();
            }
        }
    });

    clearBtn.addEventListener('click', () => {
        saveUndoState();
        initCanvas();
        canvasHint.classList.remove('hidden');
        resetResultsDisplay();
    });

    predictBtn.addEventListener('click', () => {
        predictCurrentDigit();
    });

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;
        if (e.key === 'c' || e.key === 'C') {
            clearBtn.click();
        } else if (e.key === 'p' || e.key === 'P') {
            predictBtn.click();
        } else if (e.key === 'z' || e.key === 'Z') {
            if (e.ctrlKey || e.metaKey || !e.shiftKey) {
                undoBtn.click();
            }
        } else if (e.key === 'e' || e.key === 'E') {
            eraserToolBtn.click();
        } else if (e.key === 'b' || e.key === 'B') {
            penToolBtn.click();
        }
    });

    // -------------------------------------------------------------
    // 10-Class Probability Bars Initialization
    // -------------------------------------------------------------
    function initProbabilityBars() {
        probBarsList.innerHTML = '';
        for (let i = 0; i < 10; i++) {
            const row = document.createElement('div');
            row.className = 'prob-row';
            row.id = `prob-row-${i}`;

            row.innerHTML = `
                <span class="prob-digit">${i}</span>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill" id="prob-fill-${i}"></div>
                </div>
                <span class="prob-percent" id="prob-val-${i}">0.0%</span>
            `;
            probBarsList.appendChild(row);
        }
    }
    initProbabilityBars();

    // -------------------------------------------------------------
    // Prediction API Call
    // -------------------------------------------------------------
    async function predictCurrentDigit() {
        if (!checkCanvasHasDrawing()) {
            resetResultsDisplay();
            return;
        }

        // Export canvas to base64 PNG
        const dataUrl = canvas.toDataURL('image/png');

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataUrl })
            });

            if (!res.ok) {
                throw new Error(`Server returned ${res.status}`);
            }

            const data = await res.json();
            renderPredictionResults(data);
        } catch (err) {
            console.error('Prediction error:', err);
            confidenceVal.textContent = 'Lỗi kết nối';
            confidencePill.className = 'confidence-pill low';
        }
    }

    function renderPredictionResults(data) {
        if (data.is_empty || data.prediction === null) {
            resetResultsDisplay();
            return;
        }

        // Predicted Digit with pop animation
        predictedDigitDisplay.textContent = data.prediction;
        predictedDigitDisplay.classList.remove('pop');
        void predictedDigitDisplay.offsetWidth; // trigger reflow
        predictedDigitDisplay.classList.add('pop');

        // Confidence badge & styling
        const conf = data.confidence;
        confidenceVal.textContent = `${conf.toFixed(1)}% tin cậy`;
        if (conf >= 80) {
            confidencePill.className = 'confidence-pill high';
        } else if (conf >= 50) {
            confidencePill.className = 'confidence-pill med';
        } else {
            confidencePill.className = 'confidence-pill low';
        }

        // Latency
        latencyVal.textContent = `${data.latency_ms} ms`;

        // Preprocessed Vision Image
        if (data.processed_image) {
            processedImgPreview.src = data.processed_image;
        }

        // Probability Bars
        data.probabilities.forEach((prob, idx) => {
            const fill = document.getElementById(`prob-fill-${idx}`);
            const text = document.getElementById(`prob-val-${idx}`);
            const row = document.getElementById(`prob-row-${idx}`);

            if (fill && text && row) {
                fill.style.width = `${prob}%`;
                text.textContent = `${prob.toFixed(1)}%`;

                if (idx === data.prediction) {
                    row.classList.add('winner');
                } else {
                    row.classList.remove('winner');
                }
            }
        });

        // Top Candidates Pills
        if (data.top_candidates && data.top_candidates.length > 0) {
            topCandidatesList.innerHTML = '';
            data.top_candidates.slice(0, 3).forEach((cand, rank) => {
                const pill = document.createElement('div');
                pill.className = 'candidate-pill';
                pill.innerHTML = `
                    <span class="cand-digit">#${rank + 1}: Số ${cand.digit}</span>
                    <span class="cand-val">${cand.percent}%</span>
                `;
                topCandidatesList.appendChild(pill);
            });
        }
    }

    function resetResultsDisplay() {
        predictedDigitDisplay.textContent = '?';
        confidenceVal.textContent = 'Chờ nét vẽ...';
        confidencePill.className = 'confidence-pill';
        latencyVal.textContent = '0.0 ms';
        processedImgPreview.src = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='56' height='56' viewBox='0 0 56 56'%3E%3Crect width='56' height='56' fill='%2311131b'/%3E%3C/svg%3E";

        for (let i = 0; i < 10; i++) {
            const fill = document.getElementById(`prob-fill-${i}`);
            const text = document.getElementById(`prob-val-${i}`);
            const row = document.getElementById(`prob-row-${i}`);
            if (fill) fill.style.width = '0%';
            if (text) text.textContent = '0.0%';
            if (row) row.classList.remove('winner');
        }

        topCandidatesList.innerHTML = '<span class="candidate-placeholder">Chưa có kết quả dự đoán</span>';
    }

    // -------------------------------------------------------------
    // Quick Samples Drawer (Load MNIST Test Digits)
    // -------------------------------------------------------------
    async function loadSamples() {
        try {
            const res = await fetch('/api/samples');
            const data = await res.json();
            if (data.samples && data.samples.length > 0) {
                renderSamples(data.samples);
            }
        } catch (e) {
            console.error('Failed to load samples:', e);
            samplesContainer.innerHTML = '<div class="samples-loading">Không thể tải mẫu MNIST</div>';
        }
    }

    function renderSamples(samples) {
        samplesContainer.innerHTML = '';
        samples.forEach(sample => {
            const chip = document.createElement('div');
            chip.className = 'sample-chip';
            chip.title = `Số thật từ MNIST Test: ${sample.label}`;

            const img = document.createElement('img');
            img.src = sample.image_url;
            img.alt = `Mẫu số ${sample.label}`;

            const tag = document.createElement('span');
            tag.className = 'sample-label-tag';
            tag.textContent = sample.label;

            chip.appendChild(img);
            chip.appendChild(tag);

            chip.addEventListener('click', () => {
                loadSampleOntoCanvas(sample);
            });

            samplesContainer.appendChild(chip);
        });
    }

    function loadSampleOntoCanvas(sample) {
        saveUndoState();
        canvasHint.classList.add('hidden');

        // Draw sample onto canvas centered and scaled up
        const img = new Image();
        img.onload = () => {
            initCanvas();
            // MNIST sample is 28x28 or upscaled. Draw centered at 180x180
            const size = 180;
            const offset = (CANVAS_SIZE - size) / 2;
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(img, offset, offset, size, size);

            // Trigger prediction
            predictCurrentDigit();
        };
        img.src = sample.raw_url || sample.image_url;
    }

    loadSamples();
});
