document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const uploadBtn = document.getElementById('upload-btn');
    const progressSection = document.getElementById('progress-section');
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    const resultsSection = document.getElementById('results-section');
    const errorSection = document.getElementById('error-section');
    const errorMessage = document.getElementById('error-message');
    const resultVideo = document.getElementById('result-video');
    const emotionResults = document.getElementById('emotion-results');
    
    let resultId = null;
    let pollInterval = null;
    let emotionChart = null;
    let emotionData = [];
    let currentActiveRow = null;

    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(uploadForm);
        const videoFile = formData.get('video');
        
        if (!videoFile || videoFile.size === 0) {
            showError('Please select a video file to upload.');
            return;
        }
        
        // Disable the upload button and show progress section
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Uploading...';
        progressSection.classList.remove('d-none');
        resultsSection.classList.add('d-none');
        errorSection.classList.add('d-none');
        
        // Send the video to the server
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Failed to upload video');
                });
            }
            return response.json();
        })
        .then(data => {
            console.log('Upload successful', data);
            resultId = data.result_id;
            resultVideo.src = `/uploads/${data.filename}`;
            
            // Start polling for results
            startPolling();
        })
        .catch(error => {
            console.error('Upload error:', error);
            showError(error.message || 'An error occurred during upload');
            resetUploadForm();
        });
    });
    
    function startPolling() {
        // Reset progress
        updateProgress(0);
        
        // Start polling for results
        pollInterval = setInterval(pollResults, 2000);
    }
    
    function pollResults() {
        if (!resultId) {
            clearInterval(pollInterval);
            return;
        }
        
        fetch(`/results/${resultId}`)
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to get results');
                }
                return response.json();
            })
            .then(data => {
                console.log('Poll results:', data);
                
                if (data.status === 'error') {
                    clearInterval(pollInterval);
                    showError(data.error || 'An error occurred during processing');
                    resetUploadForm();
                    return;
                }
                
                // Update progress
                updateProgress(data.progress);
                
                if (data.status === 'completed') {
                    clearInterval(pollInterval);
                    displayResults(data);
                    resetUploadForm();
                }
            })
            .catch(error => {
                console.error('Polling error:', error);
                clearInterval(pollInterval);
                showError(error.message || 'An error occurred while checking results');
                resetUploadForm();
            });
    }
    
    function updateProgress(progress) {
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
        progressText.textContent = `${progress}% complete`;
    }
    
    function displayResults(data) {
        // Show results section
        progressSection.classList.add('d-none');
        resultsSection.classList.remove('d-none');
        
        // Clear previous results
        emotionResults.innerHTML = '';
        
        // Store emotion data for timeline sync
        emotionData = data.results;
        
        // Prepare data for chart
        const faceEmotionCounts = {
            'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 
            'surprise': 0, 'neutral': 0, 'no face detected': 0
        };
        
        const voiceEmotionCounts = {
            'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0, 'sad': 0, 
            'surprise': 0, 'neutral': 0, 'no audio detected': 0
        };
        
        // Create table header with both face and voice columns
        const headerRow = document.createElement('tr');
        headerRow.innerHTML = `
            <th>Second</th>
            <th>Face Emotion</th>
            <th>Voice Emotion</th>
            <th>Actions</th>
        `;
        emotionResults.appendChild(headerRow);
        
        // Generate result rows
        data.results.forEach(result => {
            const faceEmotion = result.face_emotion.dominant_emotion;
            const voiceEmotion = result.voice_emotion.dominant_emotion;
            
            const row = document.createElement('tr');
            row.id = `second-${result.second}`;
            row.innerHTML = `
                <td>${result.second}</td>
                <td class="emotion-${faceEmotion}">${faceEmotion}</td>
                <td class="emotion-${voiceEmotion}">${voiceEmotion}</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary" type="button" data-bs-toggle="collapse" 
                            data-bs-target="#details-${result.second}" aria-expanded="false">
                        Show Details
                    </button>
                    <button class="btn btn-sm btn-outline-secondary seek-btn" data-time="${result.second}">
                        Jump to Time
                    </button>
                    <div class="collapse mt-2" id="details-${result.second}">
                        <div class="card card-body">
                            <h6>Face Emotions:</h6>
                            ${getEmotionDetailsHTML(result.face_emotion.emotions)}
                            <h6 class="mt-3">Voice Emotions:</h6>
                            ${getEmotionDetailsHTML(result.voice_emotion.emotions)}
                        </div>
                    </div>
                </td>
            `;
            emotionResults.appendChild(row);
            
            // Update emotion counts for charts
            if (faceEmotion in faceEmotionCounts) {
                faceEmotionCounts[faceEmotion]++;
            }
            
            if (voiceEmotion in voiceEmotionCounts) {
                voiceEmotionCounts[voiceEmotion]++;
            }
        });
        
        // Setup event listeners for seek buttons
        document.querySelectorAll('.seek-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const timeToSeek = parseInt(this.getAttribute('data-time'));
                resultVideo.currentTime = timeToSeek;
                resultVideo.play();
            });
        });
        
        // Create or update charts
        createEmotionCharts(faceEmotionCounts, voiceEmotionCounts);
        
        // Set up video timeupdate event for timeline sync
        resultVideo.addEventListener('timeupdate', syncTimelineWithVideo);
        
        // Seek to the beginning of the video
        resultVideo.currentTime = 0;
    }
    
    function syncTimelineWithVideo() {
        // Get the current video time (rounded to nearest second)
        const currentTime = Math.floor(resultVideo.currentTime);
        
        // Find the corresponding row in the results table
        const targetRow = document.getElementById(`second-${currentTime}`);
        
        // If we have a row for this second
        if (targetRow) {
            // Remove highlighting from previously active row
            if (currentActiveRow) {
                currentActiveRow.classList.remove('table-active');
            }
            
            // Highlight the current row
            targetRow.classList.add('table-active');
            currentActiveRow = targetRow;
            
            // Scroll the row into view if it's not visible
            const container = document.getElementById('emotion-timeline');
            const rowPosition = targetRow.offsetTop;
            const containerHeight = container.clientHeight;
            const scrollPosition = container.scrollTop;
            
            if (rowPosition < scrollPosition || rowPosition > scrollPosition + containerHeight) {
                container.scrollTop = rowPosition - containerHeight / 2;
            }
        }
    }
    
    function getEmotionDetailsHTML(emotions) {
        if (!emotions || Object.keys(emotions).length === 0) {
            return 'No detailed emotion data available';
        }
        
        let html = '<ul class="list-group">';
        for (const [emotion, score] of Object.entries(emotions)) {
            html += `<li class="list-group-item d-flex justify-content-between align-items-center">
                        <span>${emotion}</span>
                        <span class="badge bg-primary rounded-pill">${(score * 100).toFixed(2)}%</span>
                     </li>`;
        }
        html += '</ul>';
        return html;
    }
    
    function createEmotionChart(emotionCounts) {
        const ctx = document.getElementById('emotion-chart').getContext('2d');
        
        // Destroy previous chart if it exists
        if (emotionChart) {
            emotionChart.destroy();
        }
        
        const colors = {
            'angry': '#dc3545',
            'disgust': '#20c997',
            'fear': '#6610f2',
            'happy': '#28a745',
            'sad': '#6c757d',
            'surprise': '#fd7e14',
            'neutral': '#007bff'
        };
        
        emotionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(emotionCounts),
                datasets: [{
                    label: 'Emotion Distribution',
                    data: Object.values(emotionCounts),
                    backgroundColor: Object.keys(emotionCounts).map(emotion => colors[emotion] || '#17a2b8'),
                    borderColor: Object.keys(emotionCounts).map(emotion => colors[emotion] || '#17a2b8'),
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Frequency (seconds)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Emotion'
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.raw} seconds`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    function createEmotionCharts(faceEmotionCounts, voiceEmotionCounts) {
        // Clear previous charts
        document.getElementById('charts-container').innerHTML = `
            <div class="row mb-4">
                <div class="col-md-6">
                    <h5 class="text-center">Face Emotion Distribution</h5>
                    <div class="d-flex justify-content-center mb-2">
                        <div class="btn-group btn-group-sm" role="group">
                            <button type="button" class="btn btn-outline-primary active" data-chart-type="bar" data-chart-target="face">Bar</button>
                            <button type="button" class="btn btn-outline-primary" data-chart-type="pie" data-chart-target="face">Pie</button>
                        </div>
                    </div>
                    <div style="height: 300px;">
                        <canvas id="face-emotion-chart"></canvas>
                    </div>
                </div>
                <div class="col-md-6">
                    <h5 class="text-center">Voice Emotion Distribution</h5>
                    <div class="d-flex justify-content-center mb-2">
                        <div class="btn-group btn-group-sm" role="group">
                            <button type="button" class="btn btn-outline-primary active" data-chart-type="bar" data-chart-target="voice">Bar</button>
                            <button type="button" class="btn btn-outline-primary" data-chart-type="pie" data-chart-target="voice">Pie</button>
                        </div>
                    </div>
                    <div style="height: 300px;">
                        <canvas id="voice-emotion-chart"></canvas>
                    </div>
                </div>
            </div>
        `;
        
        // Color palette for emotions
        const colors = {
            'angry': '#dc3545',      // red
            'disgust': '#20c997',    // teal
            'fear': '#6610f2',       // purple
            'happy': '#28a745',      // green
            'sad': '#6c757d',        // gray
            'surprise': '#fd7e14',   // orange
            'neutral': '#007bff',    // blue
            'no face detected': '#17a2b8',  // cyan
            'no audio detected': '#17a2b8'  // cyan
        };
        
        // Store chart instances for later reference
        let faceChart = null;
        let voiceChart = null;

        // Filter out emotions with zero counts
        const faceLabels = Object.keys(faceEmotionCounts).filter(emotion => faceEmotionCounts[emotion] > 0);
        const faceData = faceLabels.map(emotion => faceEmotionCounts[emotion]);
        const faceColors = faceLabels.map(emotion => colors[emotion] || '#17a2b8');
        
        const voiceLabels = Object.keys(voiceEmotionCounts).filter(emotion => voiceEmotionCounts[emotion] > 0);
        const voiceData = voiceLabels.map(emotion => voiceEmotionCounts[emotion]);
        const voiceColors = voiceLabels.map(emotion => colors[emotion] || '#17a2b8');

        // Create face emotion chart
        const faceCtx = document.getElementById('face-emotion-chart').getContext('2d');
        faceChart = createChart(faceCtx, 'bar', faceLabels, faceData, faceColors, 'Face Emotions');

        // Create voice emotion chart
        const voiceCtx = document.getElementById('voice-emotion-chart').getContext('2d');
        voiceChart = createChart(voiceCtx, 'bar', voiceLabels, voiceData, voiceColors, 'Voice Emotions');

        // Add event listeners for chart type switchers
        document.querySelectorAll('[data-chart-type]').forEach(button => {
            button.addEventListener('click', function() {
                const chartType = this.getAttribute('data-chart-type');
                const target = this.getAttribute('data-chart-target');
                
                // Update active button state
                this.parentNode.querySelectorAll('.btn').forEach(btn => btn.classList.remove('active'));
                this.classList.add('active');
                
                // Update chart
                if (target === 'face') {
                    faceChart.destroy();
                    faceChart = createChart(faceCtx, chartType, faceLabels, faceData, faceColors, 'Face Emotions');
                } else {
                    voiceChart.destroy();
                    voiceChart = createChart(voiceCtx, chartType, voiceLabels, voiceData, voiceColors, 'Voice Emotions');
                }
            });
        });
    }

    // Helper function to create charts
    function createChart(ctx, type, labels, data, colors, title) {
        const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: type === 'pie',
                    position: 'bottom'
                },
                title: {
                    display: true,
                    text: title
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            const total = context.chart.data.datasets[0].data.reduce((a, b) => a + b, 0);
                            const percentage = ((value / total) * 100).toFixed(1);
                            return type === 'pie' 
                                ? `${label}: ${value} seconds (${percentage}%)` 
                                : `${value} seconds (${percentage}%)`;
                        }
                    }
                }
            }
        };

        // Add specific options based on chart type
        if (type === 'bar') {
            options.scales = {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Seconds'
                    }
                }
            };
        }

        return new Chart(ctx, {
            type: type,
            data: {
                labels: labels,
                datasets: [{
                    label: 'Seconds',
                    data: data,
                    backgroundColor: colors,
                    borderColor: type === 'pie' ? 'white' : colors,
                    borderWidth: type === 'pie' ? 2 : 1
                }]
            },
            options: options
        });
    }
    
    function showError(message) {
        errorSection.classList.remove('d-none');
        errorMessage.textContent = message;
    }
    
    function resetUploadForm() {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = 'Upload and Analyze';
    }
});