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

    // Color palette for emotions (moved to higher scope)
    const colors = {
        'angry': '#dc3545',      // red
        'disgust': '#20c997',    // teal
        'fear': '#6610f2',       // purple
        'happy': '#28a745',      // green
        'sad': '#6c757d',        // gray
        'surprise': '#fd7e14',   // orange
        'neutral': '#007bff',    // blue
        'no face detected': '#17a2b8',  // cyan
        'AUs Detected': '#007bff', // blue (like neutral)
        'No AUs': '#6c757d', // gray (like sad/neutral)
        'error': '#ffc107', // yellow/warning for general face errors
        'no audio detected': '#17a2b8', // cyan
        'model error': '#ffc107',       // yellow/warning for model errors
        'analysis error': '#ffc107',   // yellow/warning for analysis errors
        'prediction error': '#ffc107' // yellow/warning for prediction errors
    };

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
            const analysis = data.text_depression_analysis;
            if (analysis.error) {
                textAnalysisLabelElement.textContent = 'Error';
                textAnalysisDetailsElement.textContent = analysis.error;
                textAnalysisErrorElement.textContent = ''; // Clear specific error if general one shown
            } else if (analysis.label && analysis.score !== undefined) {
                textAnalysisLabelElement.textContent = `Label: ${analysis.label}`;
                let details = `Score: ${analysis.score.toFixed(3)}`;
                if (analysis.all_scores) {
                    details += "<br>All Scores: ";
                    analysis.all_scores.forEach(s => {
                        details += `${s.label}: ${s.score.toFixed(3)}, `;
                    });
                    details = details.slice(0, -2); // remove last comma and space
                }
                textAnalysisDetailsElement.innerHTML = details; // Use innerHTML for <br>
                textAnalysisErrorElement.textContent = '';
            } else {
                 textAnalysisLabelElement.textContent = 'No text analysis result.';
                 textAnalysisDetailsElement.textContent = '';
                 textAnalysisErrorElement.textContent = '';
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
        voiceEmotionCounts['no audio extracted'] = 0; // Corrected key
        voiceEmotionCounts['analysis error'] = 0; // Add error counts
        faceEmotionCounts['analysis error'] = 0; // Add error counts for face too

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
        emotionData.forEach(result => { // emotionData is data.results
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
                <td class="emotion-${faceEmotion.replace(/\\s+/g, '-').toLowerCase()}">${faceEmotion}</td>
                <td class="emotion-${voiceEmotion.replace(/\\s+/g, '-').toLowerCase()}">${voiceEmotion}</td>
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
    
    function getEmotionDetailsHTML(emotions) {
        if (!emotions || Object.keys(emotions).length === 0) {
            // Check if it's an error string like "no face detected"
            if (typeof emotions === 'string') return `<p>${emotions}</p>`;
            return '<p>No detailed emotion data.</p>';
        }
        let html = '<ul class="list-unstyled">';
        for (const [emotion, score] of Object.entries(emotions)) {
            html += `<li>${emotion}: ${score !== undefined ? score.toFixed(2) : 'N/A'}%</li>`;
        }
        html += '</ul>';
        return html;
    }

    function getAcousticFeaturesHTML(features) {
        if (!features || features.error) {
            return `<p>${features?.error || 'No acoustic data.'}</p>`;
        }
        // Filter out 'error' and 'second' if they exist at top level
        const featuresToShow = { ...features };
        delete featuresToShow.error;
        delete featuresToShow.second;

        if (Object.keys(featuresToShow).length === 0) {
            return '<p>No acoustic features extracted.</p>';
        }

        let html = '<ul class="list-unstyled">';
        for (const [feature, value] of Object.entries(featuresToShow)) {
            html += `<li>${feature.replace(/_/g, ' ')}: ${value !== null && value !== undefined ? parseFloat(value).toFixed(3) : 'N/A'}</li>`;
        }
        html += '</ul>';
        return html;
    }

    function createEmotionCharts(faceEmotionCounts, voiceEmotionCounts) {
        const chartsContainer = document.getElementById('charts-container');
        // Clear previous charts explicitly
        chartsContainer.innerHTML = `
            <div class="row">
                <div class="col-md-4">
                    <h5>Face Emotion Distribution</h5>
                    <canvas id="faceEmotionChart"></canvas>
                </div>
                <div class="col-md-4">
                    <h5>Voice Emotion Distribution</h5>
                    <canvas id="voiceEmotionChart"></canvas>
                </div>
                <div class="col-md-4">
                    <h5>Depression Score Timeline (Smoothed F+V)</h5>
                    <canvas id="depressionChart"></canvas>
                </div>
            </div>
        `;

        if (faceChartInstance) faceChartInstance.destroy();
        faceChartInstance = createChart(
            document.getElementById('faceEmotionChart').getContext('2d'), 
            'pie', 
            Object.keys(faceEmotionCounts), 
            Object.values(faceEmotionCounts), 
            Object.keys(faceEmotionCounts).map(label => colors[label] || '#cccccc'), // Fallback color
            'Face Emotion Distribution'
        );

        if (voiceChartInstance) voiceChartInstance.destroy();
        voiceChartInstance = createChart(
            document.getElementById('voiceEmotionChart').getContext('2d'), 
            'pie', 
            Object.keys(voiceEmotionCounts), 
            Object.values(voiceEmotionCounts), 
            Object.keys(voiceEmotionCounts).map(label => colors[label] || '#cccccc'), // Fallback color
            'Voice Emotion Distribution'
        );
        
        // Depression Timeline Chart
        if (depressionChartInstance) depressionChartInstance.destroy();
        const depressionScores = emotionData.map(r => r.final_depression_score); // Use the smoothed and scaled score
        const depressionLabels = emotionData.map((_, i) => `Sec ${i}`);

        depressionChartInstance = createChart(
            document.getElementById('depressionChart').getContext('2d'),
            'line',
            depressionLabels,
            [{
                label: 'Smoothed Depression Score (F+V)',
                data: depressionScores,
                borderColor: '#007bff',
                tension: 0.1,
                fill: false
            }],
            null,
            'Depression Score Timeline'
        );
    }

    function createChart(ctx, type, labels, data, chartColors, title) {
        // For line charts, data might be an array of datasets. For pie, it's direct values.
        let datasets;
        let backgroundColors;

        if (type === 'line') {
            datasets = data; // Expects data to be [{ label: '...', data: [], ...}]
        } else { // Pie, Bar, etc.
            datasets = [{
                label: title,
                data: data,
                backgroundColor: chartColors,
                borderColor: chartColors.map(color => chroma(color).darken().hex()), // Example border
                borderWidth: 1
            }];
            backgroundColors = chartColors; // Used by pie chart directly
        }

        return new Chart(ctx, {
            type: type,
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: (type === 'pie' || type === 'doughnut') ? 'top' : 'bottom',
                        display: !(type === 'line' && datasets.length <=1 && !datasets[0].label) // Hide legend if line chart with one unnamed dataset
                    },
                    title: {
                        display: true,
                        text: title
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null && type ==='line') {
                                     label += context.parsed.y.toFixed(2);
                                } else if (context.parsed !== null && (type === 'pie' || type === 'doughnut')) {
                                    label += context.parsed;
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: (type === 'line' || type === 'bar') ? { // Only add scales for chart types that use them
                    y: {
                        beginAtZero: true,
                        // SuggestedMin and Max can be added if a fixed scale is desired
                        // suggestedMax: (type === 'line' && title.includes("Score")) ? 100 : undefined 
                    }
                } : {}
            }
        });
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorSection.classList.remove('d-none');
        progressSection.classList.add('d-none'); // Hide progress on error
        // resultsSection.classList.add('d-none'); // Optionally hide results too
    }

    function resetUploadForm() {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = 'Upload Video';
        uploadForm.reset(); 
        // currentResultId = null; // Keep currentResultId if we want to re-poll or view old results
        // currentVideoFilename = null; 
        // Don't hide progress section immediately, might show final error message there
    }
});