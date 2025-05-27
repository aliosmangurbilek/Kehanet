// MNIST Digit Classifier Frontend Logic

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const uploadForm = document.getElementById('upload-form');
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const originalPreview = document.getElementById('original-preview');
    const processedPreview = document.getElementById('processed-preview');
    const predictionContainer = document.getElementById('prediction-container');
    const probabilityBars = document.getElementById('probability-bars');
    const predictionDigit = document.getElementById('prediction-digit');
    const loader = document.getElementById('loader');
    const howItWorksBtn = document.getElementById('how-it-works-btn');
    const howItWorksPanel = document.getElementById('how-it-works-panel');

    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });

    // Toggle How It Works panel
    if (howItWorksBtn && howItWorksPanel) {
        howItWorksBtn.addEventListener('click', function() {
            if (howItWorksPanel.classList.contains('d-none')) {
                howItWorksPanel.classList.remove('d-none');
                howItWorksBtn.textContent = 'Hide How It Works';
            } else {
                howItWorksPanel.classList.add('d-none');
                howItWorksBtn.textContent = 'How It Works';
            }
        });
    }

    // Handle file upload and drag & drop
    if (uploadArea) {
        uploadArea.addEventListener('click', function() {
            fileInput.click();
        });

        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', function() {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');

            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelected();
            }
        });

        fileInput.addEventListener('change', handleFileSelected);
    }

    // Handle file selection
    function handleFileSelected() {
        if (!fileInput.files || !fileInput.files[0]) return;

        const file = fileInput.files[0];

        // Display the original image preview
        const reader = new FileReader();
        reader.onload = function(e) {
            originalPreview.src = e.target.result;
            originalPreview.classList.remove('d-none');
        };
        reader.readAsDataURL(file);

        // Submit the form automatically when a file is selected
        if (uploadForm) {
            submitForm();
        }
    }

    // Handle form submission
    function submitForm() {
        // Show loading indicator
        if (loader) loader.style.display = 'block';

        // Clear previous results
        if (processedPreview) processedPreview.classList.add('d-none');
        if (predictionContainer) predictionContainer.classList.add('d-none');
        if (probabilityBars) probabilityBars.innerHTML = '';

        // Prepare form data for submission
        const formData = new FormData(uploadForm);

        // Send POST request to backend
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide loading indicator
            if (loader) loader.style.display = 'none';

            // Display processed image
            if (processedPreview && data.processed_image) {
                processedPreview.src = data.processed_image;
                processedPreview.classList.remove('d-none');
            }

            // Show prediction container
            if (predictionContainer) predictionContainer.classList.remove('d-none');

            // Display digit probabilities as horizontal bars
            if (probabilityBars && data.probabilities) {
                displayProbabilities(data.probabilities);
            }

            // Show top prediction
            if (predictionDigit && typeof data.prediction !== 'undefined') {
                predictionDigit.textContent = data.prediction;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            if (loader) loader.style.display = 'none';
            alert('Error processing image: ' + error.message);
        });
    }

    // Display probability bars
    function displayProbabilities(probabilities) {
        // Find maximum probability for highlighting
        const maxProb = Math.max(...probabilities);
        const maxIndex = probabilities.indexOf(maxProb);

        // Create and animate bars for each digit
        probabilityBars.innerHTML = '';
        probabilities.forEach((prob, index) => {
            const percentage = (prob * 100).toFixed(2);

            // Create bar container
            const digitContainer = document.createElement('div');
            digitContainer.className = 'digit-probability';

            // Create digit label
            const digitLabel = document.createElement('div');
            digitLabel.className = 'digit-label';
            digitLabel.textContent = index;

            // Create bar
            const bar = document.createElement('div');
            bar.className = 'probability-bar';
            if (index === maxIndex) {
                bar.classList.add('top-prediction');
            }

            // Create probability value label
            const probValue = document.createElement('div');
            probValue.className = 'probability-value';
            probValue.textContent = percentage + '%';

            // Add elements to container
            digitContainer.appendChild(digitLabel);
            digitContainer.appendChild(bar);
            digitContainer.appendChild(probValue);
            probabilityBars.appendChild(digitContainer);

            // Animate the bar width with a small delay based on index
            setTimeout(() => {
                bar.style.width = percentage + '%';
            }, 50 * index);
        });
    }

    // Handle form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            submitForm();
        });
    }
});
