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
    const transcriptTextElement = document.getElementById('transcript-text');
    const transcriptErrorElement = document.getElementById('transcript-error');
    const textAnalysisLabelElement = document.getElementById('text-analysis-label');
    const textAnalysisDetailsElement = document.getElementById('text-analysis-details');
    const textAnalysisErrorElement = document.getElementById('text-analysis-error');
    
    let currentResultId = null;
    let currentVideoFilename = null;
    let pollInterval = null;
    let faceChartInstance = null;
    let voiceChartInstance = null;
    let depressionChartInstance = null;
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
            currentResultId = data.result_id;
            currentVideoFilename = data.filename;
            
            // Clear previous results display immediately
            clearPreviousResults(); 

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
        updateProgress(0, "Initializing...");
        
        // Start polling for results
        pollInterval = setInterval(pollResults, 2000);
    }
    
    function pollResults() {
        if (!currentResultId) {
            clearInterval(pollInterval);
            return;
        }
        
        fetch(`/results/${currentResultId}`)
            .then(response => {
                if (response.status === 404) {
                    throw new Error('Analysis ID not found. Please try uploading again.');
                }
                if (response.status === 202) { // Initializing status
                    return response.json().then(data => { 
                        console.log('Poll status: Initializing...');
                        updateProgress(0, data.message || 'Initializing...'); 
                        return null; // Indicate not ready yet
                    });
                }
                if (!response.ok) {
                    return response.json().then(data => { 
                         throw new Error(data.message || 'Failed to get results');
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data === null) return; // Still initializing
                
                console.log('Poll results:', data);
                
                if (data.status === 'error') {
                    clearInterval(pollInterval);
                    showError(data.error || 'An error occurred during processing');
                    resetUploadForm();
                    // Still display partial results if available
                    if (data.results && data.results.length > 0) {
                         displayResults(data); 
                    }
                    return;
                }
                
                // Update progress bar and message
                updateProgress(data.progress, data.message);
                
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
    
    function updateProgress(progress, message) {
        progressBar.style.width = `${progress}%`;
        progressBar.setAttribute('aria-valuenow', progress);
        progressBar.textContent = `${progress}%`; // Show percentage inside bar
        progressText.textContent = message || `${progress}% complete`; // Show detailed message below
    }

    function clearPreviousResults() {
         // Clear dynamic content areas
         emotionResults.innerHTML = ''; 
         transcriptTextElement.textContent = 'Awaiting analysis...';
         transcriptErrorElement.textContent = '';
         textAnalysisLabelElement.textContent = 'Awaiting analysis...';
         textAnalysisDetailsElement.textContent = '';
         textAnalysisErrorElement.textContent = '';
         document.getElementById('charts-container').innerHTML = ''; // Clear charts
         if (resultVideo) resultVideo.src = ''; // Clear video source
         const scoreBar = document.getElementById('depression-score-bar');
         const scoreValue = document.getElementById('depression-score-value');
         const interpretation = document.getElementById('depression-interpretation');
         scoreBar.style.width = '0%';
         scoreBar.setAttribute('aria-valuenow', 0);
         scoreBar.textContent = '0';
         scoreValue.textContent = '0/100';
         interpretation.textContent = 'Awaiting analysis results...';
         interpretation.className = 'alert alert-info';
         scoreBar.className = 'progress-bar';
         
         // Destroy old chart instances if they exist
         if(faceChartInstance) faceChartInstance.destroy();
         if(voiceChartInstance) voiceChartInstance.destroy();
         if(depressionChartInstance) depressionChartInstance.destroy();
         faceChartInstance = null;
         voiceChartInstance = null;
         depressionChartInstance = null;
    }
    
    function displayResults(data) {
        // Show results section, hide progress
        progressSection.classList.add('d-none');
        resultsSection.classList.remove('d-none');
        
        // Clear previous dynamic content (safer than relying on clearPreviousResults)
        emotionResults.innerHTML = ''; 
        
        // Store data for timeline sync
        emotionData = data.results || []; // Handle case where results might be missing
        
        // Set video source using stored ID and filename
        if (currentResultId && currentVideoFilename) {
             resultVideo.src = `/uploads/${currentResultId}/${currentVideoFilename}`;
        } else {
             console.error("Missing resultId or videoFilename for setting video source");
             // Optionally hide or show an error for the video preview
        }
        
        // Display overall depression score
        const score = data.overall_depression_score !== undefined ? Math.round(data.overall_depression_score) : null;
        const scoreBar = document.getElementById('depression-score-bar');
        const scoreValue = document.getElementById('depression-score-value');
        const interpretation = document.getElementById('depression-interpretation');

        if (score !== null) {
            scoreBar.style.width = `${score}%`;
            scoreBar.setAttribute('aria-valuenow', score);
            scoreBar.textContent = score;
            scoreValue.textContent = `${score}/100`;
            scoreBar.classList.remove('bg-success', 'bg-warning', 'bg-danger');
            interpretation.classList.remove('alert-info', 'alert-success', 'alert-warning', 'alert-danger');
            if (score < 30) {
                scoreBar.classList.add('bg-success');
                interpretation.classList.add('alert-success');
                interpretation.textContent = 'Low likelihood of depression detected. The analysis suggests predominantly positive or neutral states.';
            } else if (score < 60) {
                scoreBar.classList.add('bg-warning');
                interpretation.classList.add('alert-warning');
                interpretation.textContent = 'Moderate indicators of depression detected. The analysis suggests mixed states with some concerning patterns.';
            } else {
                scoreBar.classList.add('bg-danger');
                interpretation.classList.add('alert-danger');
                interpretation.textContent = 'High indicators of depression detected. The analysis suggests patterns strongly associated with depression.';
            }
        } else {
             scoreBar.style.width = '0%';
             scoreBar.setAttribute('aria-valuenow', 0);
             scoreBar.textContent = 'N/A';
             scoreValue.textContent = 'N/A';
             interpretation.textContent = 'Overall score could not be calculated.';
             interpretation.className = 'alert alert-secondary';
             scoreBar.className = 'progress-bar bg-secondary';
        }

        // Display Transcript
        transcriptErrorElement.textContent = ''; // Clear previous error
        if (data.transcript_error) {
            transcriptTextElement.textContent = 'Transcription failed.';
            transcriptErrorElement.textContent = `Error: ${data.transcript_error}`;
        } else if (data.transcript) {
            transcriptTextElement.textContent = data.transcript;
        } else {
            transcriptTextElement.textContent = 'No transcript generated or available.';
        }

        // Display Text Analysis
        textAnalysisLabelElement.textContent = 'N/A';
        textAnalysisDetailsElement.textContent = '';
        textAnalysisErrorElement.textContent = '';
        if (data.text_depression_analysis) {
            if (data.text_depression_analysis.error) {
                textAnalysisLabelElement.textContent = 'Analysis Error';
                textAnalysisErrorElement.textContent = `Error: ${data.text_depression_analysis.error}`;
            } else {
                textAnalysisLabelElement.textContent = data.text_depression_analysis.dominant_label || 'N/A';
                // Display detailed scores (optional)
                let detailsHtml = 'Detailed Scores: ';
                if(data.text_depression_analysis.all_scores) {
                    detailsHtml += Object.entries(data.text_depression_analysis.all_scores)
                        .map(([label, score]) => `${label}: ${(score * 100).toFixed(1)}%`)
                        .join(', ');
                     textAnalysisDetailsElement.innerHTML = detailsHtml;
                }
            }
        } else {
            textAnalysisLabelElement.textContent = 'Not performed';
        }
        
        // Prepare data for charts (Face/Voice Emotion Counts)
        const faceEmotionCounts = {};
        const voiceEmotionCounts = {};
        const validEmotionLabels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'];

        validEmotionLabels.forEach(label => { 
            faceEmotionCounts[label] = 0;
            voiceEmotionCounts[label] = 0; 
        });
        faceEmotionCounts['no face detected'] = 0;
        voiceEmotionCounts['no audio extracted'] = 0;
        voiceEmotionCounts['analysis error'] = 0; // Add error counts
        faceEmotionCounts['analysis error'] = 0;

        // Create table header
        const headerRow = document.createElement('tr');
        headerRow.innerHTML = `
            <th>Second</th>
            <th>Face Emotion</th>
            <th>Voice Emotion</th>
            <th>Depression Score (F+V, Smoothed)</th> 
            <th>Actions</th>
        `;
        emotionResults.appendChild(headerRow);
        
        // Generate result rows for timeline
        emotionData.forEach(result => {
            const faceEmotion = result.face_emotion?.dominant_emotion || 'no data';
            const voiceEmotion = result.voice_emotion?.dominant_emotion || 'no data';
            // Use final_depression_score which is the smoothed+scaled FV score for the timeline
            const depressionScore = result.final_depression_score !== undefined ? 
                Math.round(result.final_depression_score) : 'N/A';
            
            let depressionScoreClass = '';
            if (depressionScore !== 'N/A') {
                if (depressionScore < 30) depressionScoreClass = 'text-success';
                else if (depressionScore < 60) depressionScoreClass = 'text-warning';
                else depressionScoreClass = 'text-danger';
            }
            
            const row = document.createElement('tr');
            row.id = `second-${result.second}`;
            // Updated details to show raw score and acoustics
            row.innerHTML = `
                <td>${result.second}</td>
                <td class="emotion-${faceEmotion}">${faceEmotion}</td>
                <td class="emotion-${voiceEmotion}">${voiceEmotion}</td>
                <td class="${depressionScoreClass}">${depressionScore}</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary" type="button" data-bs-toggle="collapse" 
                            data-bs-target="#details-${result.second}" aria-expanded="false">
                        Details
                    </button>
                    <button class="btn btn-sm btn-outline-secondary seek-btn" data-time="${result.second}">
                        Seek
                    </button>
                    <div class="collapse mt-2" id="details-${result.second}">
                        <div class="card card-body">
                            <h6>Face Details:</h6>
                            ${getEmotionDetailsHTML(result.face_emotion?.emotions)}
                            <h6 class="mt-3">Voice Details:</h6>
                            ${getEmotionDetailsHTML(result.voice_emotion?.emotions)}
                            <h6 class="mt-3">Acoustic Features:</h6>
                            ${getAcousticFeaturesHTML(result.voice_acoustic_features)}
                            <h6 class="mt-3">Analysis:</h6>
                            <p class="mb-1">Raw F+V Score: ${result.raw_depression_score_fv !== undefined ? result.raw_depression_score_fv.toFixed(3) : 'N/A'}</p>
                            <p>Smoothed & Scaled F+V Score: <strong class="${depressionScoreClass}">${depressionScore}/100</strong></p>
                        </div>
                    </div>
                </td>
            `;
            emotionResults.appendChild(row);
            
            // Update emotion counts for charts
            if (faceEmotion in faceEmotionCounts) faceEmotionCounts[faceEmotion]++;
            else if (faceEmotion !== 'no data') faceEmotionCounts['analysis error']++; 

            if (voiceEmotion in voiceEmotionCounts) voiceEmotionCounts[voiceEmotion]++;
            else if (voiceEmotion !== 'no data') voiceEmotionCounts['analysis error']++; 

        });
        
        // Setup event listeners for seek buttons
        document.querySelectorAll('.seek-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const timeToSeek = parseInt(this.getAttribute('data-time'));
                if (!isNaN(timeToSeek)) {
                    resultVideo.currentTime = timeToSeek;
                    resultVideo.play().catch(e => console.error("Video play error:", e)); // Handle potential play errors
                }
            });
        });
        
        // Create or update charts
        createEmotionCharts(faceEmotionCounts, voiceEmotionCounts);
        
        // Set up video timeupdate event for timeline sync
        resultVideo.removeEventListener('timeupdate', syncTimelineWithVideo); // Remove previous listener
        resultVideo.addEventListener('timeupdate', syncTimelineWithVideo);
        
        // Seek to the beginning and try to play (might require user interaction)
        resultVideo.currentTime = 0;
        // resultVideo.play().catch(e => console.log("Video playback requires user interaction."));
    }
    
    function syncTimelineWithVideo() {
        if (!resultVideo) return;
        const currentTime = Math.floor(resultVideo.currentTime);
        const targetRow = document.getElementById(`second-${currentTime}`);
        
        if (targetRow) {
            if (currentActiveRow && currentActiveRow !== targetRow) {
                currentActiveRow.classList.remove('table-active');
            }
            if (!targetRow.classList.contains('table-active')) {
                targetRow.classList.add('table-active');
                // Scroll into view smoothly
                targetRow.scrollIntoView({
                    behavior: 'smooth', 
                    block: 'nearest', 
                    inline: 'nearest'
                });
            }
            currentActiveRow = targetRow;
        }
    }
    
    // Updated to handle missing data
    function getEmotionDetailsHTML(emotions) {
        if (!emotions || typeof emotions !== 'object' || Object.keys(emotions).length === 0) {
            return '<p class="text-muted small">No detailed emotion data available.</p>';
        }
        
        let html = '<ul class="list-group list-group-flush">';
        for (const [emotion, score] of Object.entries(emotions)) {
             // Ensure score is a number before formatting
             const scoreNum = parseFloat(score);
             const displayScore = !isNaN(scoreNum) ? `${(scoreNum * 100).toFixed(1)}%` : 'N/A';
            html += `<li class="list-group-item d-flex justify-content-between align-items-center py-1 px-0">
                        <small>${emotion}</small>
                        <span class="badge bg-secondary rounded-pill">${displayScore}</span>
                     </li>`;
        }
        html += '</ul>';
        return html;
    }

    // NEW: Helper for Acoustic Features
    function getAcousticFeaturesHTML(features) {
         if (!features || typeof features !== 'object' || Object.keys(features).length === 0) {
            return '<p class="text-muted small">No acoustic feature data available.</p>';
        }
        let html = '<ul class="list-group list-group-flush">';
         for (const [feature, value] of Object.entries(features)) {
            const displayValue = (value !== null && !isNaN(parseFloat(value))) ? parseFloat(value).toFixed(3) : 'N/A';
            html += `<li class="list-group-item d-flex justify-content-between align-items-center py-1 px-0">
                        <small>${feature.replace(/_/g, ' ').replace(/\\b(\\w)/g, s => s.toUpperCase())}</small> 
                        <span class="badge bg-light text-dark rounded-pill">${displayValue}</span>
                     </li>`;
        }
        html += '</ul>';
        return html;
    }
    
    
    function createEmotionCharts(faceEmotionCounts, voiceEmotionCounts) {
        // Clear previous chart container content
        const chartsContainer = document.getElementById('charts-container');
        chartsContainer.innerHTML = `
            <div class="row mb-4">
                <div class="col-md-12">
                    <h5 class="text-center">Depression Score Trend (Face + Voice, Smoothed)</h5>
                    <div style="height: 200px;">
                        <canvas id="depression-trend-chart"></canvas>
                    </div>
                </div>
            </div>
            <div class="row mb-4">
                <div class="col-md-6">
                    <h5 class="text-center">Face Emotion Distribution</h5>
                     <div style="height: 300px;">
                        <canvas id="face-emotion-chart"></canvas>
                    </div>
                </div>
                <div class="col-md-6">
                    <h5 class="text-center">Voice Emotion Distribution</h5>
                    <div style="height: 300px;">
                        <canvas id="voice-emotion-chart"></canvas>
                    </div>
                </div>
            </div>
        `; // Removed chart type toggle buttons for simplicity
        
        // Destroy old chart instances
        if (faceChartInstance) faceChartInstance.destroy();
        if (voiceChartInstance) voiceChartInstance.destroy();
        if (depressionChartInstance) depressionChartInstance.destroy();

        // Color palette for emotions
        const colors = {
            'angry': '#dc3545', 'disgust': '#20c997', 'fear': '#6610f2', 'happy': '#28a745',
            'sad': '#6c757d', 'surprise': '#fd7e14', 'neutral': '#007bff',
            'no face detected': '#adb5bd', 'no audio extracted': '#adb5bd', 'analysis error': '#ffc107' // Gray/Yellow for errors
        };

        // Filter out emotions with zero counts for pie charts
        const faceLabels = Object.keys(faceEmotionCounts).filter(emotion => faceEmotionCounts[emotion] > 0);
        const faceData = faceLabels.map(emotion => faceEmotionCounts[emotion]);
        const faceColors = faceLabels.map(emotion => colors[emotion] || '#17a2b8');
        
        const voiceLabels = Object.keys(voiceEmotionCounts).filter(emotion => voiceEmotionCounts[emotion] > 0);
        const voiceData = voiceLabels.map(emotion => voiceEmotionCounts[emotion]);
        const voiceColors = voiceLabels.map(emotion => colors[emotion] || '#17a2b8');

        // Create face emotion chart (using Pie)
        const faceCtx = document.getElementById('face-emotion-chart').getContext('2d');
        if (faceLabels.length > 0) {
            faceChartInstance = createChart(faceCtx, 'pie', faceLabels, faceData, faceColors, 'Face Emotions');
        } else {
             faceCtx.font = "16px Arial"; faceCtx.textAlign = "center"; faceCtx.fillText("No face data to display", faceCtx.canvas.width / 2, faceCtx.canvas.height / 2);
        }

        // Create voice emotion chart (using Pie)
        const voiceCtx = document.getElementById('voice-emotion-chart').getContext('2d');
         if (voiceLabels.length > 0) {
            voiceChartInstance = createChart(voiceCtx, 'pie', voiceLabels, voiceData, voiceColors, 'Voice Emotions');
         } else {
             voiceCtx.font = "16px Arial"; voiceCtx.textAlign = "center"; voiceCtx.fillText("No voice data to display", voiceCtx.canvas.width / 2, voiceCtx.canvas.height / 2);
         }
        
        // Create depression trend chart
        // Use final_depression_score (smoothed F+V) for the trend line
        const depressionCtx = document.getElementById('depression-trend-chart')?.getContext('2d');
        if (depressionCtx && emotionData && emotionData.length > 0 && emotionData.some(item => item.final_depression_score !== undefined)) {
            const depressionLabels = emotionData.map(item => item.second);
            // Map to score, using null for missing values so Chart.js creates gaps
            const depressionDataPoints = emotionData.map(item => item.final_depression_score !== undefined ? item.final_depression_score : null);
            
            const gradient = depressionCtx.createLinearGradient(0, 0, 0, 200);
            gradient.addColorStop(0, 'rgba(220, 53, 69, 0.6)');    // Red (high score)
            gradient.addColorStop(0.5, 'rgba(255, 193, 7, 0.6)');  // Yellow (mid score)
            gradient.addColorStop(1, 'rgba(40, 167, 69, 0.6)');     // Green (low score)
            
            depressionChartInstance = new Chart(depressionCtx, {
                type: 'line',
                data: {
                    labels: depressionLabels,
                    datasets: [{
                        label: 'Depression Score (F+V, Smoothed)',
                        data: depressionDataPoints,
                        borderColor: 'rgba(0, 123, 255, 0.8)', // Blue line
                        backgroundColor: gradient,
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1, // Less tension for potentially gappy data
                        spanGaps: true // Connect line across null data points
                    }]
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
                                text: 'Score (0-100)'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Time (seconds)'
                            },
                            ticks: {
                                maxTicksLimit: 15, // Adjust based on typical video length
                                callback: function(val, index) {
                                     // Show every 5th label or adapt based on length
                                     const tickFrequency = Math.max(1, Math.ceil(depressionLabels.length / 15));
                                     return index % tickFrequency === 0 ? this.getLabelForValue(val) : '';
                                }
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            mode: 'index', // Show tooltip for all points at index
                            intersect: false,
                            callbacks: {
                                label: function(context) {
                                    const score = context.parsed.y;
                                    return score !== null ? `Score: ${Math.round(score)}/100` : 'Score: N/A';
                                }
                            }
                        }
                    },
                    interaction: {
                        mode: 'nearest', // Changed from index to nearest
                        axis: 'x',
                        intersect: false
                    }
                }
            });
        } else if (depressionCtx) {
             // Display message if no data for trend chart
             depressionCtx.font = "16px Arial";
             depressionCtx.textAlign = "center";
             depressionCtx.fillText("Depression trend data not available.", depressionCtx.canvas.width / 2, depressionCtx.canvas.height / 2);
        }
    }

    // Helper function to create Pie charts (Bar chart creation removed for brevity)
    function createChart(ctx, type, labels, data, colors, title) {
        // Ensure type is 'pie' for this simplified example
        if (type !== 'pie') type = 'pie'; 

        const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { padding: 15 }
                },
                title: {
                    display: false, // Title is now above the canvas
                    text: title
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const label = context.label || '';
                            const value = context.raw || 0;
                            const total = context.chart.data.datasets[0].data.reduce((a, b) => a + b, 0);
                            const percentage = total > 0 ? ((value / total) * 100).toFixed(1) : 0;
                            return `${label}: ${value}s (${percentage}%)`; 
                        }
                    }
                }
            }
        };

        return new Chart(ctx, {
            type: type,
            data: {
                labels: labels,
                datasets: [{
                    label: 'Seconds', // This label isn't prominent in pie charts
                    data: data,
                    backgroundColor: colors,
                    borderColor: 'white',
                    borderWidth: 1
                }]
            },
            options: options
        });
    }
    
    function showError(message) {
        errorSection.classList.remove('d-none');
        errorMessage.textContent = message;
        // Hide progress if showing error
        progressSection.classList.add('d-none'); 
    }
    
    function resetUploadForm() {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = 'Upload and Analyze';
        // Optionally clear the file input
        // uploadForm.reset(); 
    }
});