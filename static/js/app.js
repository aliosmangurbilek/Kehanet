// MNIST Digit Classifier - App Logic

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const drawingCanvas = document.getElementById('drawing-canvas');
    const clearBtn = document.getElementById('clear-btn');
    const predictUploadBtn = document.getElementById('predict-upload-btn');
    const predictDrawBtn = document.getElementById('predict-draw-btn');
    const originalPreview = document.getElementById('original-preview');
    const processedPreview = document.getElementById('processed-preview');
    const predictionResult = document.getElementById('prediction-result');
    const resultsSection = document.getElementById('results-section');
    const loadingOverlay = document.getElementById('loading-overlay');

    // Initialize Chart.js
    let probabilityChart = null;

    // Canvas drawing variables
    let isDrawing = false;
    let ctx = drawingCanvas.getContext('2d');
    let lastX = 0;
    let lastY = 0;

    // Initialize canvas
    function initCanvas() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, drawingCanvas.width, drawingCanvas.height);
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = 'black';
    }

    // Clear the canvas
    function clearCanvas() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, drawingCanvas.width, drawingCanvas.height);
    }

    // Drawing event listeners
    if (drawingCanvas) {
        initCanvas();

        drawingCanvas.addEventListener('mousedown', startDrawing);
        drawingCanvas.addEventListener('mousemove', draw);
        drawingCanvas.addEventListener('mouseup', stopDrawing);
        drawingCanvas.addEventListener('mouseout', stopDrawing);

        // Touch support
        drawingCanvas.addEventListener('touchstart', handleTouchStart);
        drawingCanvas.addEventListener('touchmove', handleTouchMove);
        drawingCanvas.addEventListener('touchend', stopDrawing);

        if (clearBtn) {
            clearBtn.addEventListener('click', clearCanvas);
        }

        if (predictDrawBtn) {
            predictDrawBtn.addEventListener('click', function() {
                const imageData = drawingCanvas.toDataURL('image/png');
                predictFromCanvas(imageData);
            });
        }
    }

    // File upload handling
    if (uploadArea && fileInput) {
        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');

            if (e.dataTransfer.files.length) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length) {
                handleFileSelect(fileInput.files[0]);
            }
        });

        if (predictUploadBtn) {
            predictUploadBtn.addEventListener('click', function() {
                if (fileInput.files.length) {
                    const file = fileInput.files[0];
                    const reader = new FileReader();

                    reader.onload = function(e) {
                        predictFromUpload(e.target.result);
                    };

                    reader.readAsDataURL(file);
                }
            });
        }
    }

    // Handle file selection
    function handleFileSelect(file) {
        if (!file.type.match('image.*')) {
            alert('Please select an image file.');
            return;
        }

        const reader = new FileReader();

        reader.onload = function(e) {
            // Show preview
            originalPreview.src = e.target.result;

            // Enable predict button
            if (predictUploadBtn) {
                predictUploadBtn.disabled = false;
            }
        };

        reader.readAsDataURL(file);
    }

    // Drawing functions
    function startDrawing(e) {
        isDrawing = true;
        [lastX, lastY] = getCoordinates(e);
    }

    function draw(e) {
        if (!isDrawing) return;

        // Prevent scrolling on mobile
        e.preventDefault();

        const [x, y] = getCoordinates(e);

        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.stroke();

        [lastX, lastY] = [x, y];
    }

    function stopDrawing() {
        isDrawing = false;
    }

    function getCoordinates(e) {
        const rect = drawingCanvas.getBoundingClientRect();
        const scaleX = drawingCanvas.width / rect.width;
        const scaleY = drawingCanvas.height / rect.height;

        if (e.touches && e.touches[0]) {
            return [
                (e.touches[0].clientX - rect.left) * scaleX,
                (e.touches[0].clientY - rect.top) * scaleY
            ];
        }

        return [
            (e.clientX - rect.left) * scaleX,
            (e.clientY - rect.top) * scaleY
        ];
    }

    // Touch event handlers
    function handleTouchStart(e) {
        e.preventDefault();
        if (e.touches && e.touches.length === 1) {
            startDrawing(e);
        }
    }

    function handleTouchMove(e) {
        e.preventDefault();
        if (e.touches && e.touches.length === 1) {
            draw(e);
        }
    }

    // Prediction functions
    function predictFromUpload(imageData) {
        showLoading();

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ file: imageData })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            updateResults(data);
            hideLoading();
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error processing image. Please try again.');
            hideLoading();
        });
    }

    function predictFromCanvas(imageData) {
        showLoading();

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ canvas: imageData })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            updateResults(data);
            hideLoading();
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error processing image. Please try again.');
            hideLoading();
        });
    }

    // Update results in the UI
    function updateResults(data) {
        // Update images
        originalPreview.src = data.original;
        processedPreview.src = data.preprocessed;

        // Update prediction
        predictionResult.textContent = data.prediction;

        // Update chart
        updateProbabilityChart(data.probs);

        // Show results section
        resultsSection.classList.remove('d-none');

        // Scroll to results if needed
        if (window.innerHeight < 800) {
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
    }

    // Initialize and update probability chart
    function updateProbabilityChart(probabilities) {
        const ctx = document.getElementById('probability-chart').getContext('2d');
        const labels = Array.from({length: 10}, (_, i) => i.toString());
        const maxProb = Math.max(...probabilities);
        const maxIndex = probabilities.indexOf(maxProb);

        // Format probabilities as percentages
        const formattedProbs = probabilities.map(p => (p * 100).toFixed(2));

        // Create background colors array (highlight the max)
        const backgroundColors = probabilities.map((_, index) =>
            index === maxIndex ? 'rgba(25, 135, 84, 0.8)' : 'rgba(13, 110, 253, 0.8)'
        );

        // Create border colors array
        const borderColors = probabilities.map((_, index) =>
            index === maxIndex ? 'rgb(25, 135, 84)' : 'rgb(13, 110, 253)'
        );

        // Destroy previous chart if it exists
        if (probabilityChart) {
            probabilityChart.destroy();
        }

        // Create new chart
        probabilityChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Probability (%)',
                    data: formattedProbs,
                    backgroundColor: backgroundColors,
                    borderColor: borderColors,
                    borderWidth: 1,
                    borderRadius: 5
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Probability: ${context.raw}%`;
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Probability (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Digit'
                        }
                    }
                },
                animation: {
                    duration: 1000
                }
            }
        });
    }

    // Loading indicator functions
    function showLoading() {
        if (loadingOverlay) {
            loadingOverlay.classList.remove('d-none');
        }
    }

    function hideLoading() {
        if (loadingOverlay) {
            loadingOverlay.classList.add('d-none');
        }
    }
});
