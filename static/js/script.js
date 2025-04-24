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
        
        // Display overall depression score if available
        if (data.overall_depression_score !== undefined) {
            const score = Math.round(data.overall_depression_score);
            const scoreBar = document.getElementById('depression-score-bar');
            const scoreValue = document.getElementById('depression-score-value');
            const interpretation = document.getElementById('depression-interpretation');
            
            // Update depression score gauge
            scoreBar.style.width = `${score}%`;
            scoreBar.setAttribute('aria-valuenow', score);
            scoreBar.textContent = score;
            scoreValue.textContent = `${score}/100`;
            
            // Set appropriate color based on score
            scoreBar.classList.remove('bg-success', 'bg-warning', 'bg-danger');
            if (score < 30) {
                scoreBar.classList.add('bg-success');
                interpretation.classList.remove('alert-info', 'alert-warning', 'alert-danger');
                interpretation.classList.add('alert-success');
                interpretation.textContent = 'Low likelihood of depression detected. The subject appears to display predominantly positive or neutral emotional states.';
            } else if (score < 60) {
                scoreBar.classList.add('bg-warning');
                interpretation.classList.remove('alert-info', 'alert-success', 'alert-danger');
                interpretation.classList.add('alert-warning');
                interpretation.textContent = 'Moderate indicators of depression detected. The subject displays mixed emotional states, with some concerning patterns.';
            } else {
                scoreBar.classList.add('bg-danger');
                interpretation.classList.remove('alert-info', 'alert-success', 'alert-warning');
                interpretation.classList.add('alert-danger');
                interpretation.textContent = 'High indicators of depression detected. The subject displays emotional patterns strongly associated with depression.';
            }
            
            // Add analysis of trend if score trend data is available
            if (data.depression_score_trend && data.depression_score_trend.length > 0) {
                const trend = analyzeTrend(data.depression_score_trend);
                const trendElement = document.createElement('p');
                trendElement.classList.add('mt-2');
                
                if (trend.direction === 'increasing') {
                    trendElement.innerHTML = `<strong>Trend Analysis:</strong> Depression indicators are <span class="text-danger">increasing</span> over time, suggesting worsening mood.`;
                } else if (trend.direction === 'decreasing') {
                    trendElement.innerHTML = `<strong>Trend Analysis:</strong> Depression indicators are <span class="text-success">decreasing</span> over time, suggesting improving mood.`;
                } else {
                    trendElement.innerHTML = `<strong>Trend Analysis:</strong> Depression indicators are <span class="text-primary">stable</span> over time.`;
                }
                
                interpretation.appendChild(trendElement);
            }
        }
        
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
            <th>Depression Score</th>
            <th>Actions</th>
        `;
        emotionResults.appendChild(headerRow);
        
        // Generate result rows
        data.results.forEach(result => {
            const faceEmotion = result.face_emotion.dominant_emotion;
            const voiceEmotion = result.voice_emotion.dominant_emotion;
            const depressionScore = result.depression_score !== undefined ? 
                Math.round(result.depression_score) : 'N/A';
            
            // Determine depression score class for styling
            let depressionScoreClass = '';
            let depressionLabel = '';
            if (depressionScore !== 'N/A') {
                if (depressionScore < 30) {
                    depressionScoreClass = 'text-success';
                    depressionLabel = 'Low';
                } else if (depressionScore < 60) {
                    depressionScoreClass = 'text-warning';
                    depressionLabel = 'Moderate';
                } else {
                    depressionScoreClass = 'text-danger';
                    depressionLabel = 'High';
                }
            }
            
            const row = document.createElement('tr');
            row.id = `second-${result.second}`;
            row.innerHTML = `
                <td>${result.second}</td>
                <td class="emotion-${faceEmotion}">${faceEmotion}</td>
                <td class="emotion-${voiceEmotion}">${voiceEmotion}</td>
                <td class="${depressionScoreClass}">${depressionScore} <span class="badge ${getBadgeClass(depressionScore)}">${depressionLabel}</span></td>
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
                            ${result.depression_score !== undefined ? `
                            <h6 class="mt-3">Depression Analysis:</h6>
                            <p>Score for this moment: <strong class="${depressionScoreClass}">${depressionScore}/100</strong> (${depressionLabel})</p>
                            ` : ''}
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
        createEmotionCharts(faceEmotionCounts, voiceEmotionCounts, data.depression_score_trend);
        
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
    
    function createEmotionCharts(faceEmotionCounts, voiceEmotionCounts, depressionTrend) {
        // Clear previous charts
        document.getElementById('charts-container').innerHTML = `
            <div class="row mb-4">
                <div class="col-md-12">
                    <h5 class="text-center">Depression Score Over Time</h5>
                    <div style="height: 200px;">
                        <canvas id="depression-trend-chart"></canvas>
                    </div>
                </div>
            </div>
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
        
        // Create enhanced depression trend chart using trend data if available, otherwise fallback to per-second data
        const depressionScores = depressionTrend || (emotionData && emotionData.length > 0 ? 
            emotionData.map(item => item.depression_score) : null);
            
        if (depressionScores && depressionScores.length > 0) {
            const depressionCtx = document.getElementById('depression-trend-chart').getContext('2d');
            
            // Create timeline labels based on data length
            const depressionLabels = Array.from({ length: depressionScores.length }, (_, i) => `Second ${i}`);
            
            // Create gradient for depression chart
            const gradient = depressionCtx.createLinearGradient(0, 0, 0, 200);
            gradient.addColorStop(0, 'rgba(220, 53, 69, 0.8)');    // Red for high
            gradient.addColorStop(0.5, 'rgba(255, 193, 7, 0.8)');  // Yellow for mid
            gradient.addColorStop(1, 'rgba(40, 167, 69, 0.8)');    // Green for low
            
            // Calculate linear regression for trendline
            const trend = analyzeTrend(depressionScores);
            const trendLineData = depressionScores.map((_, i) => {
                // y = mx + b formula where b is y-intercept
                // For first point: scores[0] = b (since x=0)
                // For any point: y = slope*x + first_score
                return trend.slope * i + depressionScores[0];
            });
            
            new Chart(depressionCtx, {
                type: 'line',
                data: {
                    labels: depressionLabels,
                    datasets: [
                        {
                            label: 'Depression Score',
                            data: depressionScores,
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: gradient,
                            borderWidth: 2,
                            fill: true,
                            tension: 0.4
                        },
                        {
                            label: 'Trend Line',
                            data: trendLineData,
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            fill: false,
                            pointRadius: 0,
                            tension: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Depression Score'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Video Timeline'
                            },
                            ticks: {
                                // Display fewer x-axis labels if there are many data points
                                maxTicksLimit: 10,
                                callback: function(val, index) {
                                    // Show fewer labels when we have a lot of seconds
                                    return index % Math.ceil(depressionLabels.length / 10) === 0 ? this.getLabelForValue(val) : '';
                                }
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    if (context.dataset.label === 'Trend Line') {
                                        return `Trend: ${Math.round(context.raw)}/100`;
                                    }
                                    return `Depression: ${Math.round(context.raw)}/100`;
                                }
                            }
                        },
                        annotation: {
                            annotations: {
                                thresholdHigh: {
                                    type: 'line',
                                    yMin: 60,
                                    yMax: 60,
                                    borderColor: 'rgba(220, 53, 69, 0.5)',
                                    borderWidth: 1,
                                    borderDash: [5, 5],
                                    label: {
                                        enabled: true,
                                        content: 'High',
                                        position: 'end'
                                    }
                                },
                                thresholdMod: {
                                    type: 'line',
                                    yMin: 30,
                                    yMax: 30,
                                    borderColor: 'rgba(255, 193, 7, 0.5)',
                                    borderWidth: 1,
                                    borderDash: [5, 5],
                                    label: {
                                        enabled: true,
                                        content: 'Moderate',
                                        position: 'end'
                                    }
                                }
                            }
                        }
                    },
                    interaction: {
                        mode: 'index',
                        intersect: false
                    }
                }
            });
        }
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

    // Helper function to determine badge class based on depression score
    function getBadgeClass(score) {
        if (score < 30) return 'bg-success';
        if (score < 60) return 'bg-warning';
        return 'bg-danger';
    }

    // Function to analyze trend direction in depression scores
    function analyzeTrend(scores) {
        if (!scores || scores.length < 3) {
            return { direction: 'stable', slope: 0 };
        }
        
        // Calculate linear regression
        const n = scores.length;
        const indices = Array.from({ length: n }, (_, i) => i);
        
        // Calculate means
        const meanX = indices.reduce((sum, x) => sum + x, 0) / n;
        const meanY = scores.reduce((sum, y) => sum + y, 0) / n;
        
        // Calculate slope (m) using least squares method
        let numerator = 0;
        let denominator = 0;
        
        for (let i = 0; i < n; i++) {
            numerator += (indices[i] - meanX) * (scores[i] - meanY);
            denominator += Math.pow(indices[i] - meanX, 2);
        }
        
        const slope = denominator !== 0 ? numerator / denominator : 0;
        
        // Determine trend direction based on slope and significance threshold
        const TREND_THRESHOLD = 0.5; // Threshold to consider a trend significant
        
        let direction;
        if (slope > TREND_THRESHOLD) {
            direction = 'increasing';
        } else if (slope < -TREND_THRESHOLD) {
            direction = 'decreasing';
        } else {
            direction = 'stable';
        }
        
        return { direction, slope };
    }
});