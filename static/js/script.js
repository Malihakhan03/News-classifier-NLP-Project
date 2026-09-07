/**
 * News Headline Classification - Professional Frontend JavaScript
 * Handles:
 * - Real-time AJAX classification with Model Selector
 * - Speech-to-Text Voice Dictation (Web Speech API)
 * - Explainable NLP Token Highlights
 * - Multi-Model Side-by-Side Comparison
 * - Batch File Drag-and-Drop & CSV Export
 * - Dark/Light Theme Switching
 * - Prediction History & Copy to Clipboard
 */

document.addEventListener('DOMContentLoaded', () => {

    // ========================================================
    // 1. Dark / Light Theme Toggle
    // ========================================================
    const themeToggleBtn = document.getElementById('themeToggleBtn');
    const themeIcon = document.getElementById('themeIcon');
    const htmlElement = document.documentElement;

    const savedTheme = localStorage.getItem('nlp_theme') || 'light';
    setTheme(savedTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', () => {
            const currentTheme = htmlElement.getAttribute('data-bs-theme') || 'light';
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            setTheme(newTheme);
        });
    }

    function setTheme(theme) {
        htmlElement.setAttribute('data-bs-theme', theme);
        localStorage.setItem('nlp_theme', theme);
        if (themeIcon) {
            if (theme === 'dark') {
                themeIcon.className = 'bi bi-sun-fill text-warning';
            } else {
                themeIcon.className = 'bi bi-moon-stars-fill text-light';
            }
        }
    }


    // ========================================================
    // 2. Classifier Dashboard Elements
    // ========================================================
    const classificationForm = document.getElementById('classificationForm');
    const headlineInput = document.getElementById('headlineInput');
    const modelSelector = document.getElementById('modelSelector');
    const charCounter = document.getElementById('charCounter');
    const classifyBtn = document.getElementById('classifyBtn');
    const clearBtn = document.getElementById('clearBtn');
    const voiceMicBtn = document.getElementById('voiceMicBtn');
    const micIcon = document.getElementById('micIcon');
    const micText = document.getElementById('micText');
    const validationAlert = document.getElementById('validationAlert');
    const validationMessage = document.getElementById('validationMessage');

    const placeholderResult = document.getElementById('placeholderResult');
    const loadingResult = document.getElementById('loadingResult');
    const activeResult = document.getElementById('activeResult');
    const copyResultBtn = document.getElementById('copyResultBtn');

    const resultCategory = document.getElementById('resultCategory');
    const resultConfidence = document.getElementById('resultConfidence');
    const preprocessedTokens = document.getElementById('preprocessedTokens');
    const probabilityBars = document.getElementById('probabilityBars');
    const tokenImportanceBadges = document.getElementById('tokenImportanceBadges');
    const multiModelCards = document.getElementById('multiModelCards');

    const historyTableBody = document.getElementById('historyTableBody');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const sampleItems = document.querySelectorAll('.sample-item');

    let currentPredictionData = null;


    // Character Counter
    if (headlineInput && charCounter) {
        headlineInput.addEventListener('input', () => {
            const count = headlineInput.value.length;
            charCounter.textContent = `${count} char${count === 1 ? '' : 's'}`;
            hideValidation();
        });
    }

    // Clear Button
    if (clearBtn && headlineInput) {
        clearBtn.addEventListener('click', () => {
            headlineInput.value = '';
            charCounter.textContent = '0 chars';
            headlineInput.focus();
            hideValidation();
        });
    }

    // Sample Headlines
    sampleItems.forEach(item => {
        item.addEventListener('click', () => {
            const headline = item.getAttribute('data-headline');
            if (headline && headlineInput) {
                headlineInput.value = headline;
                charCounter.textContent = `${headline.length} chars`;
                hideValidation();
                headlineInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
                submitClassification(headline);
            }
        });
    });

    // Form Submission
    if (classificationForm) {
        classificationForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const headline = headlineInput.value.trim();
            submitClassification(headline);
        });
    }


    // ========================================================
    // 3. Speech-to-Text Voice Dictation (Web Speech API)
    // ========================================================
    if (voiceMicBtn && ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        let isListening = false;

        voiceMicBtn.addEventListener('click', () => {
            if (!isListening) {
                try {
                    recognition.start();
                    isListening = true;
                    voiceMicBtn.classList.add('listening');
                    if (micText) micText.textContent = 'Listening...';
                } catch (e) {
                    console.error('Speech recognition error:', e);
                }
            } else {
                recognition.stop();
                isListening = false;
                voiceMicBtn.classList.remove('listening');
                if (micText) micText.textContent = 'Dictate';
            }
        });

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            if (transcript && headlineInput) {
                headlineInput.value = transcript;
                charCounter.textContent = `${transcript.length} chars`;
                hideValidation();
                submitClassification(transcript);
            }
        };

        recognition.onend = () => {
            isListening = false;
            voiceMicBtn.classList.remove('listening');
            if (micText) micText.textContent = 'Dictate';
        };

        recognition.onerror = (event) => {
            console.warn('Speech recognition warning:', event.error);
            isListening = false;
            voiceMicBtn.classList.remove('listening');
            if (micText) micText.textContent = 'Dictate';
        };
    } else if (voiceMicBtn) {
        voiceMicBtn.title = 'Speech Recognition is not supported by your current browser.';
    }


    // ========================================================
    // 4. Submit Single Classification
    // ========================================================
    async function submitClassification(headline) {
        if (!headline) {
            showValidation('Please enter a news headline.');
            return;
        }

        if (headline.length < 3) {
            showValidation('Headline is too short. Please enter a complete news headline (at least 3 characters).');
            return;
        }

        if (!/[a-zA-Z]/.test(headline)) {
            showValidation('Headline must contain alphabetic words, not only numbers or symbols.');
            return;
        }

        const selectedModel = modelSelector ? modelSelector.value : 'best';

        hideValidation();
        setLoadingState(true);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    headline: headline,
                    model: selectedModel
                })
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                const errorMsg = data.error || 'Failed to classify headline. Please try again.';
                showValidation(errorMsg);
                setLoadingState(false);
                return;
            }

            currentPredictionData = data;
            displayResult(data);
            addToHistoryTable(data);
            setLoadingState(false);

        } catch (err) {
            console.error('Prediction request error:', err);
            showValidation('Network or server error occurred. Please verify backend is running.');
            setLoadingState(false);
        }
    }


    // ========================================================
    // 5. Render Prediction Results in Dashboard
    // ========================================================
    function displayResult(data) {
        if (!activeResult) return;

        resultCategory.textContent = data.predicted_category;
        if (data.is_unknown || data.confidence === null || data.confidence === undefined) {
            resultConfidence.textContent = `Uncertain / Out-of-Domain (${data.raw_confidence ? data.raw_confidence.toFixed(1) + '%' : 'Low'})`;
        } else {
            resultConfidence.textContent = `${data.confidence.toFixed(2)}%`;
        }
        preprocessedTokens.textContent = data.cleaned_tokens || '(No tokens remaining)';

        // Out-of-Domain / Unknown alert
        const unknownAlert = document.getElementById('unknownAlert');
        const unknownText = document.getElementById('unknownText');
        if (unknownAlert && unknownText) {
            if (data.is_unknown && data.rejection_reason) {
                unknownText.textContent = data.rejection_reason;
                unknownAlert.classList.remove('d-none');
            } else {
                unknownAlert.classList.add('d-none');
            }
        }

        // Top 3 Predicted Candidates
        const top3Container = document.getElementById('top3Container');
        const top3Badges = document.getElementById('top3Badges');
        if (top3Container && top3Badges) {
            top3Badges.innerHTML = '';
            if (data.top_3 && data.top_3.length > 0 && !data.is_unknown) {
                data.top_3.forEach((cand, idx) => {
                    const pill = document.createElement('span');
                    pill.className = `badge rounded-pill border px-3 py-2 ${cand.badge || 'bg-secondary'} fs-7`;
                    pill.innerHTML = `<span class="fw-bold me-1">#${idx + 1}</span> <i class="bi ${cand.icon || 'bi-tag'} me-1"></i> ${cand.category} <span class="badge bg-white text-dark ms-1">${cand.percentage}%</span>`;
                    top3Badges.appendChild(pill);
                });
                top3Container.classList.remove('d-none');
            } else {
                top3Container.classList.add('d-none');
            }
        }

        // Data-driven Uncertainty alert
        const uncertainAlert = document.getElementById('uncertainAlert');
        const uncertainText = document.getElementById('uncertainText');
        if (uncertainAlert && uncertainText) {
            if (!data.is_unknown && data.is_uncertain && data.uncertainty_note) {
                uncertainText.textContent = data.uncertainty_note;
                uncertainAlert.classList.remove('d-none');
            } else {
                uncertainAlert.classList.add('d-none');
            }
        }

        // Ambiguity / Low margin alert
        const ambiguityAlert = document.getElementById('ambiguityAlert');
        const ambiguityText = document.getElementById('ambiguityText');
        if (ambiguityAlert && ambiguityText) {
            if (!data.is_unknown && !data.is_uncertain && data.is_ambiguous && data.ambiguity_note) {
                ambiguityText.textContent = data.ambiguity_note;
                ambiguityAlert.classList.remove('d-none');
            } else {
                ambiguityAlert.classList.add('d-none');
            }
        }

        // 1. Render Explainable Token Badges (Positive vs Negative feature impact)
        if (tokenImportanceBadges) {
            tokenImportanceBadges.innerHTML = '';
            if (data.token_importance && data.token_importance.length > 0) {
                data.token_importance.forEach(item => {
                    const badge = document.createElement('span');
                    badge.className = 'token-chip shadow-xs';
                    const isPositive = item.impact === 'positive';
                    const badgeClass = isPositive ? 'bg-success bg-opacity-75' : 'bg-danger bg-opacity-75';
                    const sign = item.importance > 0 ? '+' : '';
                    badge.innerHTML = `
                        <span>${item.token}</span>
                        <span class="badge ${badgeClass} text-white fs-8">${sign}${item.importance.toFixed(3)}</span>
                    `;
                    tokenImportanceBadges.appendChild(badge);
                });
            } else {
                tokenImportanceBadges.innerHTML = '<span class="text-muted small">Features mapped via TF-IDF feature space</span>';
            }
        }

        // 2. Render Probability Bars
        probabilityBars.innerHTML = '';
        if (data.probability_distribution && data.probability_distribution.length > 0) {
            data.probability_distribution.forEach(item => {
                const isTop = item.is_top;
                const barColor = isTop ? 'bg-primary' : 'bg-secondary opacity-75';
                const textColor = isTop ? 'text-primary fw-bold' : 'text-dark';

                const barElement = document.createElement('div');
                barElement.className = 'prob-row';
                barElement.innerHTML = `
                    <div class="d-flex justify-content-between align-items-center mb-1">
                        <span class="prob-label ${textColor}">
                            ${isTop ? '<i class="bi bi-check-circle-fill text-success me-1"></i>' : ''}
                            ${item.category}
                        </span>
                        <span class="prob-pct ${textColor}">${item.percentage.toFixed(2)}%</span>
                    </div>
                    <div class="progress" role="progressbar" aria-label="${item.category}" aria-valuenow="${item.percentage}" aria-valuemin="0" aria-valuemax="100">
                        <div class="progress-bar ${barColor}" style="width: 0%"></div>
                    </div>
                `;
                probabilityBars.appendChild(barElement);

                setTimeout(() => {
                    const progressBar = barElement.querySelector('.progress-bar');
                    if (progressBar) {
                        progressBar.style.width = `${Math.max(item.percentage, 1.5)}%`;
                    }
                }, 50);
            });
        }

        // 3. Render Multi-Model Side-by-Side Cards
        if (multiModelCards && data.multi_model_comparison) {
            multiModelCards.innerHTML = '';
            for (const [mName, mInfo] of Object.entries(data.multi_model_comparison)) {
                const col = document.createElement('div');
                col.className = 'col-md-4';
                col.innerHTML = `
                    <div class="p-2 px-3 rounded-3 border bg-body-tertiary text-center h-100">
                        <span class="text-muted fs-8 text-uppercase fw-semibold d-block text-truncate">${mName}</span>
                        <span class="badge ${mInfo.badge} my-1 px-2 py-1">${mInfo.category}</span>
                        <span class="small fw-bold text-primary d-block">${mInfo.confidence}%</span>
                    </div>
                `;
                multiModelCards.appendChild(col);
            }
        }

        if (copyResultBtn) copyResultBtn.classList.remove('d-none');
        if (placeholderResult) placeholderResult.classList.add('d-none');
        if (loadingResult) loadingResult.classList.add('d-none');
        activeResult.classList.remove('d-none');
    }


    // Copy Result to Clipboard
    if (copyResultBtn) {
        copyResultBtn.addEventListener('click', () => {
            if (!currentPredictionData) return;
            const textToCopy = `News Headline: "${currentPredictionData.headline}"\nPredicted Category: ${currentPredictionData.predicted_category} (${currentPredictionData.confidence}% Confidence)\nNLP Tokens: ${currentPredictionData.cleaned_tokens}`;
            navigator.clipboard.writeText(textToCopy).then(() => {
                copyResultBtn.innerHTML = '<i class="bi bi-check-lg text-success me-1"></i> Copied!';
                setTimeout(() => {
                    copyResultBtn.innerHTML = '<i class="bi bi-clipboard me-1"></i> Copy';
                }, 2000);
            });
        });
    }


    // Add to History Table
    function addToHistoryTable(data) {
        if (!historyTableBody) return;

        const noHistoryRow = document.getElementById('noHistoryRow');
        if (noHistoryRow) noHistoryRow.remove();

        const tr = document.createElement('tr');
        const badgeClass = data.category_info ? data.category_info.badge : 'bg-secondary';

        tr.innerHTML = `
            <td class="text-muted small text-nowrap">${data.timestamp}</td>
            <td class="fw-medium text-dark">${escapeHtml(data.headline)}</td>
            <td>
                <span class="badge ${badgeClass} px-2 py-1">
                    ${data.predicted_category}
                </span>
            </td>
            <td class="text-end fw-bold text-primary">${data.confidence.toFixed(2)}%</td>
        `;

        historyTableBody.insertBefore(tr, historyTableBody.firstChild);
        while (historyTableBody.children.length > 10) {
            historyTableBody.removeChild(historyTableBody.lastChild);
        }
    }

    // Clear History Button
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', async () => {
            try {
                const res = await fetch('/api/clear-history', { method: 'POST' });
                const json = await res.json();
                if (json.success && historyTableBody) {
                    historyTableBody.innerHTML = `
                        <tr id="noHistoryRow">
                            <td colspan="4" class="text-center py-4 text-muted">
                                <i class="bi bi-inbox fs-4 d-block mb-1 opacity-50"></i>
                                Prediction history cleared.
                            </td>
                        </tr>
                    `;
                }
            } catch (err) {
                console.error('Failed to clear history:', err);
            }
        });
    }


    // ========================================================
    // 6. Batch Classification Dashboard JS
    // ========================================================
    const dropZone = document.getElementById('dropZone');
    const batchFileInput = document.getElementById('batchFileInput');
    const selectedFileName = document.getElementById('selectedFileName');
    const batchTextarea = document.getElementById('batchTextarea');
    const startBatchBtn = document.getElementById('startBatchBtn');
    const batchAlert = document.getElementById('batchAlert');

    const batchPlaceholder = document.getElementById('batchPlaceholder');
    const batchLoading = document.getElementById('batchLoading');
    const batchActiveView = document.getElementById('batchActiveView');
    const batchSummaryText = document.getElementById('batchSummaryText');
    const batchDistributionPills = document.getElementById('batchDistributionPills');
    const batchResultsTable = document.getElementById('batchResultsTable');
    const batchTableBody = document.getElementById('batchTableBody');
    const batchSearchInput = document.getElementById('batchSearchInput');
    const downloadCsvBtn = document.getElementById('downloadCsvBtn');

    let batchResultsData = [];
    let selectedFileObj = null;

    if (dropZone && batchFileInput) {
        // Drag events
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.add('dragover');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                dropZone.classList.remove('dragover');
            }, false);
        });

        dropZone.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length > 0) {
                handleSelectedFile(files[0]);
            }
        });

        batchFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleSelectedFile(e.target.files[0]);
            }
        });

        function handleSelectedFile(file) {
            selectedFileObj = file;
            if (selectedFileName) {
                selectedFileName.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
                selectedFileName.classList.remove('d-none');
            }
            hideBatchAlert();
        }
    }

    if (startBatchBtn) {
        startBatchBtn.addEventListener('click', async () => {
            const activeTab = document.querySelector('#batchInputTabs .nav-link.active');
            const isUploadTab = activeTab && activeTab.id === 'upload-tab';

            let formData = new FormData();

            if (isUploadTab) {
                if (!selectedFileObj) {
                    showBatchAlert('Please select or drag a .csv or .txt file first.');
                    return;
                }
                formData.append('file', selectedFileObj);
            } else {
                const textContent = batchTextarea.value.trim();
                if (!textContent) {
                    showBatchAlert('Please paste one or more news headlines in the textarea.');
                    return;
                }
                formData.append('text_lines', textContent);
            }

            hideBatchAlert();
            setBatchLoadingState(true);

            try {
                const response = await fetch('/api/batch-predict', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (!response.ok || !data.success) {
                    showBatchAlert(data.error || 'Failed to process batch.');
                    setBatchLoadingState(false);
                    return;
                }

                batchResultsData = data.results;
                displayBatchResults(data);
                setBatchLoadingState(false);

            } catch (err) {
                console.error('Batch error:', err);
                showBatchAlert('Server connection error during batch processing.');
                setBatchLoadingState(false);
            }
        });
    }

    function displayBatchResults(data) {
        if (!batchActiveView) return;

        if (batchSummaryText) {
            batchSummaryText.textContent = `Successfully classified ${data.total_classified} headlines`;
        }

        // Render Distribution Pills
        if (batchDistributionPills && data.category_distribution) {
            batchDistributionPills.innerHTML = '';
            for (const [cat, count] of Object.entries(data.category_distribution)) {
                if (count > 0) {
                    const pill = document.createElement('div');
                    pill.className = 'p-2 px-3 bg-white rounded-3 border small shadow-xs';
                    pill.innerHTML = `<strong>${cat}:</strong> <span class="badge bg-primary ms-1">${count}</span>`;
                    batchDistributionPills.appendChild(pill);
                }
            }
        }

        // Render Table Rows
        renderBatchTableRows(data.results);

        if (downloadCsvBtn) downloadCsvBtn.classList.remove('d-none');
        if (batchPlaceholder) batchPlaceholder.classList.add('d-none');
        if (batchLoading) batchLoading.classList.add('d-none');
        batchActiveView.classList.remove('d-none');
    }

    function renderBatchTableRows(rows) {
        if (!batchTableBody) return;
        batchTableBody.innerHTML = '';

        rows.forEach(r => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="text-muted small">${r.id}</td>
                <td class="fw-medium text-dark">${escapeHtml(r.headline)}</td>
                <td><span class="badge ${r.badge} px-2 py-1">${r.predicted_category}</span></td>
                <td class="text-end fw-bold text-primary">${r.confidence}%</td>
            `;
            batchTableBody.appendChild(tr);
        });
    }

    // Filter Search in Batch Table
    if (batchSearchInput) {
        batchSearchInput.addEventListener('input', () => {
            const q = batchSearchInput.value.toLowerCase().trim();
            const filtered = batchResultsData.filter(item => 
                item.headline.toLowerCase().includes(q) || 
                item.predicted_category.toLowerCase().includes(q)
            );
            renderBatchTableRows(filtered);
        });
    }

    // Download CSV
    if (downloadCsvBtn) {
        downloadCsvBtn.addEventListener('click', () => {
            if (!batchResultsData || batchResultsData.length === 0) return;

            let csvContent = "data:text/csv;charset=utf-8,ID,Headline,Predicted_Category,Confidence_Percentage\n";
            batchResultsData.forEach(r => {
                const escapedHeadline = `"${r.headline.replace(/"/g, '""')}"`;
                csvContent += `${r.id},${escapedHeadline},${r.predicted_category},${r.confidence}\n`;
            });

            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", `classified_news_headlines_${Date.now()}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    }

    function setBatchLoadingState(isLoading) {
        if (startBatchBtn) startBatchBtn.disabled = isLoading;
        if (isLoading) {
            if (batchPlaceholder) batchPlaceholder.classList.add('d-none');
            if (batchActiveView) batchActiveView.classList.add('d-none');
            if (batchLoading) batchLoading.classList.remove('d-none');
        }
    }

    function showBatchAlert(msg) {
        if (batchAlert) {
            batchAlert.textContent = msg;
            batchAlert.classList.remove('d-none');
        }
    }

    function hideBatchAlert() {
        if (batchAlert) batchAlert.classList.add('d-none');
    }


    // ========================================================
    // 7. General Helpers
    // ========================================================
    function setLoadingState(isLoading) {
        if (classifyBtn) {
            classifyBtn.disabled = isLoading;
            if (isLoading) {
                classifyBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2" role="status"></span> Classifying...';
            } else {
                classifyBtn.innerHTML = '<i class="bi bi-lightning-charge-fill me-1"></i> <span>Classify Headline</span>';
            }
        }

        if (isLoading) {
            if (placeholderResult) placeholderResult.classList.add('d-none');
            if (activeResult) activeResult.classList.add('d-none');
            if (loadingResult) loadingResult.classList.remove('d-none');
        }
    }

    function showValidation(message) {
        if (validationAlert && validationMessage) {
            validationMessage.textContent = message;
            validationAlert.classList.remove('d-none');
        }
    }

    function hideValidation() {
        if (validationAlert) {
            validationAlert.classList.add('d-none');
        }
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
});
