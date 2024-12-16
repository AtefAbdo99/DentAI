class DentalAnalyzer {
    constructor() {
        if (!this.checkDependencies()) {
            console.error('Required dependencies are missing');
            return;
        }
        this.apiUrl = '/api';  // This will use relative path instead of hardcoded localhost
        this.cases = [];
        this.eventListeners = new Map();

        // Wait for DOM to be ready before initializing
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initialize());
        } else {
            this.initialize();
        }
    }

    initialize() {
        try {
            this.setupEventListeners();
            this.initializeDragAndDrop();
            this.initializeThemeSwitch();
            this.initializeImageComparison();
            this.setupKeyboardShortcuts();
            this.setupAdvancedControls();
            this.initializeHeader();
            
            // Initialize statistics
            this.updateAnalysisStats();
        } catch (error) {
            console.error('Error during initialization:', error);
        }
    }

    checkDependencies() {
        const required = {
            'Chart': typeof Chart !== 'undefined',
            'jsPDF': typeof window.jspdf !== 'undefined',
            'FontAwesome': document.querySelector('link[href*="font-awesome"]') !== null
        };

        const missing = Object.entries(required)
            .filter(([, exists]) => !exists)
            .map(([name]) => name);

        if (missing.length > 0) {
            this.showNotification(
                `Missing required dependencies: ${missing.join(', ')}`, 
                'error'
            );
            return false;
        }
        return true;
    }

    initializeDragAndDrop() {
        const dropZone = document.querySelector('.drag-drop-zone');
        if (!dropZone) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        dropZone.addEventListener('dragenter', () => {
            dropZone.classList.add('drag-highlight');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-highlight');
        });

        dropZone.addEventListener('drop', (e) => {
            dropZone.classList.remove('drag-highlight');
            const files = Array.from(e.dataTransfer.files);
            if (files.length > 0) {
                this.handleFiles(files);
            }
        });

        // Also handle click to browse
        dropZone.addEventListener('click', () => {
            document.getElementById('imageInput').click();
        });
    }

    setupEventListeners() {
        const listeners = [
            ['analyzeBtn', 'click', () => this.analyzeImages()],
            ['clearBtn', 'click', () => this.clearAnalysis()],
            ['exportBtn', 'click', () => this.exportReport()],
            ['imageInput', 'change', (e) => this.handleFiles(Array.from(e.target.files))]
        ];

        listeners.forEach(([id, event, handler]) => {
            const element = document.getElementById(id);
            if (element) {
                element.addEventListener(event, handler);
                this.eventListeners.set(`${id}-${event}`, { element, event, handler });
            }
        });
    }

    async handleFiles(files) {
        const validFiles = files.filter(file => {
            const validTypes = ['image/jpeg', 'image/png', 'image/jpg'];
            if (!validTypes.includes(file.type)) {
                this.showError(`File "${file.name}" is not a supported image type`);
                return false;
            }
            return true;
        });

        if (validFiles.length === 0) return;

        // Automatically start analysis when files are added
        await this.analyzeImages(validFiles);
    }

    async analyzeImages(files = null) {
        try {
            const imagesToAnalyze = files || Array.from(document.getElementById('imageInput').files);
            if (imagesToAnalyze.length === 0) {
                this.showError('Please select images to analyze');
                return;
            }

            // Show loading state
            const progressContainer = document.getElementById('progressContainer');
            const progressBar = document.getElementById('progressBar');
            const progressText = document.getElementById('progressText');
            
            if (progressContainer && progressBar) {
                progressContainer.classList.remove('d-none');
                progressBar.style.width = '0%';
                progressText.textContent = 'Starting analysis...';
            }

            for (let i = 0; i < imagesToAnalyze.length; i++) {
                try {
                    const formData = new FormData();
                    formData.append('file', imagesToAnalyze[i]);

                    const response = await fetch(`${this.apiUrl}/analyze`, {
                        method: 'POST',
                        body: formData
                    });

                    if (!response.ok) throw new Error('Analysis failed');

                    const result = await response.json();
                    this.cases.push(result);
                    this.displayResult(result, imagesToAnalyze[i]);

                    // Update progress
                    if (progressBar && progressText) {
                        const percent = ((i + 1) / imagesToAnalyze.length) * 100;
                        progressBar.style.width = `${percent}%`;
                        progressText.textContent = `Analyzing image ${i + 1} of ${imagesToAnalyze.length}`;
                    }
                } catch (error) {
                    console.error('Error analyzing image:', error);
                    this.showError(`Failed to analyze "${imagesToAnalyze[i].name}": ${error.message}`);
                }
            }

            // Hide progress after completion
            if (progressContainer) {
                setTimeout(() => {
                    progressContainer.classList.add('d-none');
                    progressBar.style.width = '0%';
                }, 500); // Small delay for better UX
            }

            // Update stats
            this.updateAnalysisStats();
        } catch (error) {
            console.error('Analysis error:', error);
            this.showError('Analysis failed: ' + error.message);
            if (progressContainer) {
                progressContainer.classList.add('d-none');
            }
        }
    }

    updateAnalysisStats() {
        try {
            // Get statistics elements
            const analysesToday = document.getElementById('analysesToday');
            const accuracyRate = document.getElementById('accuracyRate');
            const avgProcessingTime = document.getElementById('avgProcessingTime');
            const totalAnalyses = document.getElementById('totalAnalyses');

            if (!analysesToday || !accuracyRate || !avgProcessingTime || !totalAnalyses) {
                console.warn('Some statistics elements are missing');
                return;
            }

            // Calculate today's analyses
            const today = new Date().toDateString();
            const todayCount = this.cases.filter(c => 
                new Date(c.timestamp || Date.now()).toDateString() === today
            ).length;

            // Update today's analyses
            this.animateCounter(analysesToday, todayCount);

            // Calculate and update accuracy rate
            if (this.cases.length > 0) {
                const avgAccuracy = this.cases.reduce((acc, curr) => {
                    if (!curr.predictions || !curr.predictions[0]) return acc;
                    // Convert confidence to percentage (multiply by 100)
                    return acc + (parseFloat(curr.predictions[0][1]) * 100);
                }, 0) / this.cases.length;

                this.animateCounter(accuracyRate, avgAccuracy, '%');
            } else {
                this.animateCounter(accuracyRate, 0, '%');
            }

            // Calculate and update processing time
            if (this.cases.length > 0) {
                const avgTime = this.cases.reduce((acc, curr) => 
                    acc + (parseFloat(curr.processingTime) || 2.5), 0) / this.cases.length;
                this.animateCounter(avgProcessingTime, avgTime, 's');
            } else {
                this.animateCounter(avgProcessingTime, 0, 's');
            }

            // Update total analyses
            const storedTotal = parseInt(localStorage.getItem('totalAnalyses') || '0');
            const newCases = this.cases.filter(c => !c.counted).length;
            const newTotal = storedTotal + newCases;
            
            // Mark new cases as counted
            this.cases.forEach(c => c.counted = true);
            
            localStorage.setItem('totalAnalyses', newTotal.toString());
            this.animateCounter(totalAnalyses, newTotal);

            // Update trends with percentage changes
            this.updateTrends();
        } catch (error) {
            console.error('Error updating statistics:', error);
        }
    }

    animateCounter(element, target, suffix = '') {
        const start = parseFloat(element.textContent) || 0;
        const duration = 1000;
        const startTime = performance.now();

        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            const current = start + (target - start) * this.easeOutQuart(progress);
            element.textContent = current.toFixed(1) + (suffix || '');

            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };

        requestAnimationFrame(animate);
    }

    easeOutQuart(x) {
        return 1 - Math.pow(1 - x, 4);
    }

    updateTrends() {
        const updateTrend = (elementId, value, prevValue) => {
            const trendElement = document.getElementById(elementId);
            if (!trendElement) return;

            // Calculate percentage change
            const change = prevValue > 0 ? ((value - prevValue) / prevValue * 100) : 0;
            const trendIcon = trendElement.querySelector('i');
            const trendValue = trendElement.querySelector('span');

            if (trendIcon && trendValue) {
                trendIcon.className = `fas fa-arrow-${change >= 0 ? 'up' : 'down'}`;
                trendValue.textContent = `${Math.abs(change).toFixed(1)}%`;
                trendElement.className = `stat-trend ${change >= 0 ? 'positive' : 'negative'}`;
            }
        };

        const currentStats = {
            todayAnalyses: this.cases.filter(c => 
                new Date(c.timestamp || Date.now()).toDateString() === new Date().toDateString()
            ).length,
            accuracy: this.getAverageAccuracy(),
            processingTime: this.getAverageProcessingTime(),
            totalAnalyses: this.getTotalAnalyses()
        };

        const prevStats = JSON.parse(localStorage.getItem('prevStats') || '{}');

        // Update trends
        Object.entries(currentStats).forEach(([key, value]) => {
            const elementId = `${key}Trend`;
            updateTrend(elementId, value, prevStats[key] || 0);
        });

        // Save current values for next comparison
        localStorage.setItem('prevStats', JSON.stringify(currentStats));
    }

    getAverageAccuracy() {
        if (this.cases.length === 0) return 0;
        
        return this.cases.reduce((acc, curr) => {
            if (!curr.predictions || !curr.predictions[0]) return acc;
            return acc + (parseFloat(curr.predictions[0][1]) * 100);
        }, 0) / this.cases.length;
    }

    getAverageProcessingTime() {
        if (this.cases.length === 0) return 0;
        
        return this.cases.reduce((acc, curr) => 
            acc + (parseFloat(curr.processingTime) || 2.5), 0) / this.cases.length;
    }

    getTotalAnalyses() {
        return parseInt(localStorage.getItem('totalAnalyses') || '0');
    }

    displayResult(result, file) {
        try {
            const container = document.getElementById('resultsContainer');
            if (!container) {
                throw new Error('Results container not found');
            }

            // Validate result data
            if (!result || !result.predictions || !result.predictions[0]) {
                throw new Error('Invalid result data');
            }

            const card = document.createElement('div');
            card.className = 'col-md-6 mb-4';
            
            const confidence = (result.predictions[0][1] * 100).toFixed(1);
            const confidenceClass = confidence > 90 ? 'high' : confidence > 70 ? 'medium' : 'low';
            
            card.innerHTML = `
                <div class="analysis-card">
                    <div class="analysis-image-container">
                        <img src="${URL.createObjectURL(file)}" alt="Dental X-Ray" class="analysis-image">
                    </div>
                    
                    <div class="diagnosis-header">
                        <h3 class="diagnosis-title">
                            <i class="fas fa-microscope"></i>
                            ${result.predictions[0][0]}
                        </h3>
                        <div class="confidence-score ${confidenceClass}">
                            <i class="fas fa-chart-line"></i>
                            <span>${confidence}% Confidence</span>
                        </div>
                    </div>
                    
                    <div class="analysis-content">
                        <div class="findings-section">
                            <h4 class="section-header">
                                <i class="fas fa-clipboard-list"></i>
                                Clinical Findings
                            </h4>
                            ${result.findings ? result.findings.map(finding => `
                                <div class="finding-item">
                                    <i class="fas fa-check-circle"></i>
                                    <span>${finding}</span>
                                </div>
                            `).join('') : '<p>No findings available</p>'}
                        </div>
                    </div>
                </div>
            `;

            container.appendChild(card);

            // Update statistics
            this.updateAnalysisStats();

        } catch (error) {
            console.error('Error in displayResult:', error);
            this.showError(`Failed to display analysis result: ${error.message}`);
        }
    }

    clearAnalysis() {
        document.getElementById('resultsContainer').innerHTML = '';
        document.getElementById('imageInput').value = '';
        // Don't reset total analyses when clearing
        this.cases = [];
        this.updateAnalysisStats();
    }

    resetInterface() {
        // Reset drag & drop zone
        const dragDropZone = document.getElementById('dragDropZone');
        if (dragDropZone) {
            dragDropZone.classList.remove('drag-highlight');
            dragDropZone.querySelector('p').textContent = 'Drag & Drop X-ray images here or click to browse';
        }

        // Reset progress bar if exists
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            progressBar.style.width = '0%';
            progressBar.parentElement.classList.add('d-none');
        }

        // Clear any error messages
        const errorMessages = document.querySelectorAll('.error-message');
        errorMessages.forEach(msg => msg.remove());
    }

    showLoading() {
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-spinner"></div>
        `;
        document.body.appendChild(overlay);
    }

    hideLoading() {
        const overlay = document.querySelector('.loading-overlay');
        if (overlay) {
            overlay.remove();
        }
    }

    showError(message) {
        try {
            const container = document.getElementById('resultsContainer');
            if (!container) {
                console.error('Results container not found');
                return;
            }

            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.innerHTML = `
                <i class="fas fa-exclamation-circle"></i>
                <span>${message}</span>
            `;
            container.prepend(errorDiv);
            setTimeout(() => errorDiv.remove(), 5000);
        } catch (err) {
            console.error('Error showing error message:', err);
        }
    }

    async exportReport() {
        if (this.cases.length === 0) {
            this.showError('No analyses to export');
            return;
        }

        try {
            // Show loading state
            const exportBtn = document.getElementById('exportBtn');
            const originalText = exportBtn.innerHTML;
            exportBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
            exportBtn.disabled = true;

            // Initialize PDF with custom settings
            const pdf = new window.jspdf.jsPDF({
                orientation: 'portrait',
                unit: 'mm',
                format: 'a4'
            });

            // Add report header
            this.addReportHeader(pdf);
            
            // Add summary section
            this.addSummarySection(pdf);
            
            // Add each case analysis
            let currentPage = 1;
            this.cases.forEach((caseData, index) => {
                if (currentPage > 1) pdf.addPage();
                this.addCaseDetails(pdf, caseData, index + 1);
                currentPage++;
            });

            // Add footer to all pages
            const pageCount = pdf.internal.getNumberOfPages();
            for (let i = 1; i <= pageCount; i++) {
                pdf.setPage(i);
                this.addPageFooter(pdf, i, pageCount);
            }

            // Save the PDF
            const filename = `dental-analysis-report-${new Date().toISOString().split('T')[0]}.pdf`;
            pdf.save(filename);

            // Reset button state
            setTimeout(() => {
                exportBtn.innerHTML = originalText;
                exportBtn.disabled = false;
            }, 500);

            this.showNotification('Report exported successfully', 'success');
        } catch (error) {
            console.error('Export error:', error);
            this.showError('Failed to export report: ' + error.message);
            
            // Reset button state on error
            const exportBtn = document.getElementById('exportBtn');
            if (exportBtn) {
                exportBtn.innerHTML = '<i class="fas fa-file-export"></i> Export Report';
                exportBtn.disabled = false;
            }
        }
    }

    addReportHeader(pdf) {
        // Add logo and title
        pdf.setFillColor(41, 128, 185); // Professional blue color
        pdf.rect(0, 0, 210, 40, 'F');
        
        pdf.setTextColor(255);
        pdf.setFontSize(24);
        pdf.text('DentAI Pro Analysis Report', 105, 20, { align: 'center' });
        
        pdf.setFontSize(12);
        pdf.text('Advanced Dental Radiograph Analysis System', 105, 30, { align: 'center' });

        // Add report info
        pdf.setTextColor(0);
        pdf.setFontSize(10);
        const today = new Date().toLocaleDateString();
        pdf.text(`Generated: ${today}`, 20, 50);
        pdf.text(`Total Cases: ${this.cases.length}`, 20, 55);
    }

    addSummarySection(pdf) {
        let yPos = 70;

        // Add summary title
        pdf.setFillColor(245, 247, 250);
        pdf.rect(15, yPos - 5, 180, 50, 'F');
        
        pdf.setFontSize(14);
        pdf.setTextColor(41, 128, 185);
        pdf.text('Analysis Summary', 20, yPos);
        
        // Calculate and add statistics
        pdf.setFontSize(10);
        pdf.setTextColor(0);
        
        const avgConfidence = this.cases.reduce((acc, c) => 
            acc + (parseFloat(c.predictions[0][1]) * 100), 0) / this.cases.length;
        
        const diagnoses = this.cases.reduce((acc, c) => {
            const diagnosis = c.predictions[0][0];
            acc[diagnosis] = (acc[diagnosis] || 0) + 1;
            return acc;
        }, {});

        yPos += 10;
        pdf.text(`Average Confidence: ${avgConfidence.toFixed(1)}%`, 25, yPos);
        
        yPos += 10;
        pdf.text('Diagnosis Distribution:', 25, yPos);
        
        Object.entries(diagnoses).forEach(([diagnosis, count], index) => {
            yPos += 5;
            const percentage = ((count / this.cases.length) * 100).toFixed(1);
            pdf.text(`• ${diagnosis}: ${count} cases (${percentage}%)`, 30, yPos);
        });
    }

    addCaseDetails(pdf, caseData, caseNumber) {
        let yPos = 130;

        // Case header
        pdf.setFillColor(41, 128, 185, 0.1);
        pdf.rect(15, yPos - 5, 180, 15, 'F');
        
        pdf.setFontSize(12);
        pdf.setTextColor(41, 128, 185);
        pdf.text(`Case ${caseNumber}`, 20, yPos);
        
        const confidence = (caseData.predictions[0][1] * 100).toFixed(1);
        pdf.text(`Confidence: ${confidence}%`, 160, yPos);

        // Diagnosis
        yPos += 20;
        pdf.setFontSize(11);
        pdf.setTextColor(0);
        pdf.text('Diagnosis:', 20, yPos);
        pdf.setFontSize(11);
        pdf.text(caseData.predictions[0][0], 50, yPos);

        // Findings
        yPos += 10;
        pdf.setFontSize(11);
        pdf.text('Clinical Findings:', 20, yPos);
        
        caseData.findings.forEach(finding => {
            yPos += 6;
            pdf.setFontSize(10);
            pdf.text(`• ${finding}`, 25, yPos);
        });

        // Recommendations
        yPos += 15;
        pdf.setFontSize(11);
        pdf.text('Recommendations:', 20, yPos);
        
        caseData.recommendations.forEach(rec => {
            yPos += 6;
            pdf.setFontSize(10);
            pdf.text(`• ${rec}`, 25, yPos);
        });

        // Management Plan
        if (caseData.management) {
            yPos += 15;
            pdf.setFontSize(11);
            pdf.text('Management Plan:', 20, yPos);
            
            Object.entries(caseData.management).forEach(([key, value]) => {
                yPos += 6;
                pdf.setFontSize(10);
                pdf.text(`• ${key}: ${value}`, 25, yPos);
            });
        }
    }

    addPageFooter(pdf, pageNumber, totalPages) {
        pdf.setFontSize(8);
        pdf.setTextColor(128);
        
        // Add page numbers
        pdf.text(
            `Page ${pageNumber} of ${totalPages}`, 
            105, 
            287, 
            { align: 'center' }
        );
        
        // Add footer text
        pdf.text(
            '© 2024 DentAI Pro - Advanced Dental Radiograph Analysis System', 
            105, 
            292, 
            { align: 'center' }
        );
    }

    // Helper methods for enhanced styling
    addModernSection(doc, { title, content, yPos, confidence, icon }) {
        // Section background with gradient
        doc.setFillColor(245, 247, 250);
        doc.roundedRect(20, yPos, 170, 25, 3, 3, 'F');
        
        // Title with icon
        doc.setTextColor(41, 128, 185);
        doc.setFontSize(12);
        doc.text(`${icon} ${title}:`, 25, yPos + 8);
        
        // Content
        doc.setTextColor(0);
        doc.text(content, 70, yPos + 8);
        
        // Confidence bar if provided
        if (confidence) {
            const confidencePercent = (confidence * 100).toFixed(1);
            const barWidth = 140 * (confidence);
            
            // Background bar
            doc.setFillColor(220, 220, 220);
            doc.roundedRect(25, yPos + 15, 140, 4, 2, 2, 'F');
            
            // Progress bar with color based on confidence
            const color = confidence >= 0.9 ? [46, 204, 113] : 
                         confidence >= 0.7 ? [241, 196, 15] : 
                         [231, 76, 60];
            doc.setFillColor(...color);
            doc.roundedRect(25, yPos + 15, barWidth, 4, 2, 2, 'F');
            
            // Confidence text
            doc.setFontSize(10);
            doc.text(`${confidencePercent}%`, 170, yPos + 18);
        }
    }

    // Add a new method for the summary page
    addSummaryPage(doc) {
        // Header
        doc.setFillColor(41, 128, 185);
        doc.rect(0, 0, 210, 30, 'F');
        
        doc.setTextColor(255, 255, 255);
        doc.setFontSize(24);
        doc.text('Summary Report', 105, 15, { align: 'center' });
        
        let yPos = 40;

        // Summary statistics
        doc.setTextColor(0);
        doc.setFontSize(12);
        
        // Total cases analyzed
        doc.text(`Total Cases Analyzed: ${this.cases.length}`, 20, yPos);
        yPos += 15;

        // Diagnosis distribution
        const diagnoses = {};
        this.cases.forEach(case_data => {
            const diagnosis = case_data.predictions[0][0];
            diagnoses[diagnosis] = (diagnoses[diagnosis] || 0) + 1;
        });

        doc.text('Diagnosis Distribution:', 20, yPos);
        yPos += 10;

        Object.entries(diagnoses).forEach(([diagnosis, count]) => {
            const percentage = ((count / this.cases.length) * 100).toFixed(1);
            doc.text(`• ${diagnosis}: ${count} cases (${percentage}%)`, 30, yPos);
            yPos += 8;
        });

        // Add timestamp
        doc.setFontSize(8);
        doc.setTextColor(128);
        doc.text(`Report generated on ${new Date().toLocaleString()}`, 20, 280);
    }

    // Helper method to add sections
    addSection(doc, title, items, yPos) {
        const sectionHeight = Math.max(40, items.length * 8 + 15);
        doc.setFillColor(245, 247, 250);
        doc.rect(20, yPos, 170, sectionHeight, 'F');
        
        doc.setTextColor(41, 128, 185);
        doc.text(title, 25, yPos + 8);
        
        doc.setTextColor(0);
        items.forEach((item, index) => {
            const text = `• ${item}`;
            const lines = doc.splitTextToSize(text, 150);
            lines.forEach((line, lineIndex) => {
                doc.text(line, 30, yPos + 18 + (index * 8) + (lineIndex * 8));
            });
        });
        
        return yPos + sectionHeight + 10;
    }

    // Helper method for management section
    addManagementSection(doc, management, yPos) {
        const items = Object.entries(management);
        const sectionHeight = Math.max(40, items.length * 8 + 15);
        
        doc.setFillColor(245, 247, 250);
        doc.rect(20, yPos, 170, sectionHeight, 'F');
        
        doc.setTextColor(41, 128, 185);
        doc.text('Management Plan', 25, yPos + 8);
        
        doc.setTextColor(0);
        items.forEach(([key, value], index) => {
            const text = `• ${key}: ${value}`;
            const lines = doc.splitTextToSize(text, 150);
            lines.forEach((line, lineIndex) => {
                doc.text(line, 30, yPos + 18 + (index * 8) + (lineIndex * 8));
            });
        });
        
        return yPos + sectionHeight + 10;
    }

    getImageDataUrl(img) {
        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        return canvas.toDataURL('image/jpeg', 0.75);
    }

    initializeThemeSwitch() {
        const themeSwitch = document.createElement('div');
        themeSwitch.className = 'theme-switch';
        themeSwitch.innerHTML = `
            <button class="theme-switch-button" aria-label="Toggle theme">
                <i class="fas fa-moon"></i>
            </button>
        `;
        document.body.appendChild(themeSwitch);

        const button = themeSwitch.querySelector('.theme-switch-button');
        const icon = button.querySelector('i');

        // Load saved theme
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.body.dataset.theme = savedTheme;
        icon.className = savedTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';

        button.addEventListener('click', () => {
            const isDark = document.body.dataset.theme === 'dark';
            document.body.dataset.theme = isDark ? 'light' : 'dark';
            localStorage.setItem('theme', document.body.dataset.theme);
            
            // Animate icon change
            icon.style.transform = 'scale(0)';
            setTimeout(() => {
                icon.className = isDark ? 'fas fa-moon' : 'fas fa-sun';
                icon.style.transform = 'scale(1)';
            }, 150);
        });
    }

    initializeImageComparison() {
        const comparisonContainers = document.querySelectorAll('.image-comparison');
        comparisonContainers.forEach(container => {
            const slider = container.querySelector('.comparison-slider');
            let isDown = false;

            const move = (e) => {
                if (!isDown) return;
                const rect = container.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const percent = (x / rect.width) * 100;
                slider.style.left = `${Math.min(Math.max(percent, 0), 100)}%`;
            };

            slider.addEventListener('mousedown', () => isDown = true);
            window.addEventListener('mouseup', () => isDown = false);
            window.addEventListener('mousemove', move);
        });
    }

    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + A: Analyze Images
            if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
                e.preventDefault();
                this.analyzeImages();
            }
            // Ctrl/Cmd + E: Export Report
            if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
                e.preventDefault();
                this.exportReport();
            }
            // Ctrl/Cmd + C: Clear Analysis
            if ((e.ctrlKey || e.metaKey) && e.key === 'c') {
                e.preventDefault();
                this.clearAnalysis();
            }
        });
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'}"></i>
            <span>${message}</span>
        `;
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.style.opacity = '0';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    addImageControls() {
        const viewers = document.querySelectorAll('.image-viewer');
        viewers.forEach(viewer => {
            const controls = document.createElement('div');
            controls.className = 'image-controls';
            controls.innerHTML = `
                <button class="zoom-in"><i class="fas fa-search-plus"></i></button>
                <button class="zoom-out"><i class="fas fa-search-minus"></i></button>
                <button class="rotate"><i class="fas fa-redo"></i></button>
            `;
            viewer.appendChild(controls);
            this.initializeImageControls(viewer);
        });
    }

    initializeImageControls(viewer) {
        const img = viewer.querySelector('img');
        let scale = 1;
        let rotation = 0;

        viewer.querySelector('.zoom-in').onclick = () => {
            scale = Math.min(scale * 1.2, 3);
            this.updateImageTransform(img, scale, rotation);
        };

        viewer.querySelector('.zoom-out').onclick = () => {
            scale = Math.max(scale / 1.2, 0.5);
            this.updateImageTransform(img, scale, rotation);
        };

        viewer.querySelector('.rotate').onclick = () => {
            rotation += 90;
            this.updateImageTransform(img, scale, rotation);
        };
    }

    updateImageTransform(img, scale, rotation) {
        img.style.transform = `scale(${scale}) rotate(${rotation}deg)`;
    }

    addImageComparison() {
        const container = document.createElement('div');
        container.className = 'image-comparison-container';
        container.innerHTML = `
            <div class="comparison-view">
                <div class="original-image"></div>
                <div class="enhanced-image"></div>
                <div class="comparison-slider">
                    <div class="slider-handle"></div>
                </div>
            </div>
            <div class="comparison-controls">
                <button class="enhance-btn">Enhance Image</button>
                <button class="reset-btn">Reset</button>
            </div>
        `;
        
        // Add image enhancement logic
        const enhanceBtn = container.querySelector('.enhance-btn');
        enhanceBtn.addEventListener('click', () => this.enhanceImage(container));
        
        return container;
    }

    enhanceImage(container) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const originalImage = container.querySelector('.original-image img');
        
        canvas.width = originalImage.naturalWidth;
        canvas.height = originalImage.naturalHeight;
        
        // Apply image enhancements
        ctx.drawImage(originalImage, 0, 0);
        ctx.filter = 'contrast(120%) brightness(110%) sharpen(1)';
        ctx.drawImage(canvas, 0, 0);
        
        const enhancedImage = container.querySelector('.enhanced-image');
        enhancedImage.style.backgroundImage = `url(${canvas.toDataURL()})`;
    }

    setupAdvancedControls() {
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.shiftKey) {
                switch(e.key) {
                    case 'Z': // Zoom mode
                        this.setTool('zoom');
                        break;
                    case 'A': // Annotation mode
                        this.setTool('annotate');
                        break;
                    case 'M': // Measurement mode
                        this.setTool('measure');
                        break;
                    case 'R': // Reset view
                        this.resetView();
                        break;
                }
            }
        });

        // Touch gestures
        let touchStartX = 0;
        let touchStartY = 0;

        document.addEventListener('touchstart', (e) => {
            touchStartX = e.touches[0].clientX;
            touchStartY = e.touches[0].clientY;
        });

        document.addEventListener('touchend', (e) => {
            const touchEndX = e.changedTouches[0].clientX;
            const touchEndY = e.changedTouches[0].clientY;
            
            const deltaX = touchEndX - touchStartX;
            const deltaY = touchEndY - touchStartY;
            
            if (Math.abs(deltaX) > 100) {
                // Horizontal swipe
                if (deltaX > 0) {
                    this.previousCase();
                } else {
                    this.nextCase();
                }
            } else if (Math.abs(deltaY) > 100) {
                // Vertical swipe
                if (deltaY < 0) {
                    this.zoomIn();
                } else {
                    this.zoomOut();
                }
            }
        });
    }

    initializeHeader() {
        const header = document.querySelector('.header-section');
        if (header) {
            header.style.position = 'fixed';
            header.style.top = '0';
            header.style.left = '0';
            header.style.right = '0';
            header.style.zIndex = '1000';
            header.style.backgroundColor = 'var(--background-color)';
            header.style.borderBottom = '1px solid var(--border-color)';
            header.style.padding = '10px 20px';

            // Add padding to body to prevent content from hiding behind fixed header
            document.body.style.paddingTop = `${header.offsetHeight}px`;
        }

        // Initialize statistics
        this.updateStatistics();
        setInterval(() => this.updateStatistics(), 60000); // Update every minute
    }

    updateStatistics() {
        // Get statistics elements
        const todayAnalyses = document.getElementById('todayAnalyses');
        const accuracyRate = document.getElementById('accuracyRate');
        const avgProcessingTime = document.getElementById('avgProcessingTime');
        const totalAnalyses = document.getElementById('totalAnalyses');

        // Calculate today's analyses
        const today = new Date().toDateString();
        const todayCount = this.cases.filter(c => 
            new Date(c.timestamp).toDateString() === today
        ).length;
        
        // Animate counter
        this.animateCounter(todayAnalyses, todayCount);

        // Calculate accuracy
        if (this.cases.length > 0) {
            const avgAccuracy = this.cases.reduce((acc, c) => 
                acc + (parseFloat(c.predictions[0][1]) * 100), 0) / this.cases.length;
            
            this.animateCounter(accuracyRate, avgAccuracy.toFixed(1), '%');
            this.updateAccuracyRing(avgAccuracy);
        }

        // Calculate processing time
        if (this.cases.length > 0) {
            const avgTime = this.cases.reduce((acc, c) => 
                acc + (parseFloat(c.processingTime) || 2.5), 0) / this.cases.length;
            
            this.animateCounter(avgProcessingTime, avgTime.toFixed(1), 's');
            this.updateProcessingTimeChart(this.getProcessingTimeHistory());
        }

        // Update total analyses
        const total = parseInt(localStorage.getItem('totalAnalyses') || '0') + this.cases.length;
        localStorage.setItem('totalAnalyses', total.toString());
        this.animateCounter(totalAnalyses, total);

        // Update trends
        this.updateTrends();
    }

    toggleHelp() {
        const helpContent = `
            <div class="modal-content help-modal">
                <div class="modal-header">
                    <h3>Help & Documentation</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <section class="help-section">
                        <h4><i class="fas fa-upload"></i> Uploading Images</h4>
                        <p>Drag and drop your dental X-ray images into the upload area or click to browse files.</p>
                        <p>Supported formats: JPG, PNG</p>
                    </section>
                    
                    <section class="help-section">
                        <h4><i class="fas fa-keyboard"></i> Keyboard Shortcuts</h4>
                        <ul>
                            <li><kbd>Ctrl</kbd> + <kbd>A</kbd> - Analyze Images</li>
                            <li><kbd>Ctrl</kbd> + <kbd>E</kbd> - Export Report</li>
                            <li><kbd>Ctrl</kbd> + <kbd>C</kbd> - Clear Analysis</li>
                            <li><kbd>Esc</kbd> - Close Modals</li>
                        </ul>
                    </section>
                    
                    <section class="help-section">
                        <h4><i class="fas fa-brain"></i> AI Analysis</h4>
                        <p>Our advanced AI model analyzes dental X-rays to provide:</p>
                        <ul>
                            <li>Accurate diagnosis with confidence scores</li>
                            <li>Detailed clinical findings</li>
                            <li>Professional recommendations</li>
                            <li>Comprehensive management plans</li>
                        </ul>
                    </section>
                </div>
            </div>
        `;
        this.showModal(helpContent);
    }

    toggleSettings() {
        const currentTheme = localStorage.getItem('theme') || 'light';
        const autoAnalyze = localStorage.getItem('autoAnalyze') === 'true';
        const confidenceThreshold = localStorage.getItem('confidenceThreshold') || '70';

        const settingsContent = `
            <div class="modal-content settings-modal">
                <div class="modal-header">
                    <h3>Settings</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <section class="settings-section">
                        <h4>Appearance</h4>
                        <div class="setting-item">
                            <label>Theme</label>
                            <select id="themeSelect" class="form-control">
                                <option value="light" ${currentTheme === 'light' ? 'selected' : ''}>Light</option>
                                <option value="dark" ${currentTheme === 'dark' ? 'selected' : ''}>Dark</option>
                            </select>
                        </div>
                    </section>

                    <section class="settings-section">
                        <h4>Analysis Options</h4>
                        <div class="setting-item">
                            <label>
                                <input type="checkbox" id="autoAnalyze" ${autoAnalyze ? 'checked' : ''}>
                                Auto-analyze on upload
                            </label>
                        </div>
                        <div class="setting-item">
                            <label>Confidence Threshold</label>
                            <input type="range" id="confidenceThreshold" 
                                   min="0" max="100" value="${confidenceThreshold}"
                                   class="form-range">
                            <span id="confidenceValue">${confidenceThreshold}%</span>
                        </div>
                    </section>
                </div>
            </div>
        `;
        
        this.showModal(settingsContent);
        this.initializeSettingsHandlers();
    }

    showHistory() {
        const historyContent = `
            <div class="modal-content history-modal">
                <div class="modal-header">
                    <h3>Analysis History</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="history-filters">
                        <select class="form-control" id="historyFilter">
                            <option value="all">All Time</option>
                            <option value="today">Today</option>
                            <option value="week">This Week</option>
                            <option value="month">This Month</option>
                        </select>
                    </div>
                    <div class="history-list">
                        ${this.generateHistoryList()}
                    </div>
                </div>
            </div>
        `;
        
        this.showModal(historyContent);
    }

    generateHistoryList() {
        if (this.cases.length === 0) {
            return '<p class="text-center">No analysis history available</p>';
        }

        return this.cases.map((case_data, index) => `
            <div class="history-item">
                <div class="history-item-header">
                    <span class="history-date">${new Date(case_data.timestamp).toLocaleString()}</span>
                    <span class="history-confidence ${this.getConfidenceClass(case_data.predictions[0][1])}">
                        ${(case_data.predictions[0][1] * 100).toFixed(1)}%
                    </span>
                </div>
                <div class="history-item-content">
                    <h4>${case_data.predictions[0][0]}</h4>
                    <p>Findings: ${case_data.findings.length}</p>
                    <p>Recommendations: ${case_data.recommendations.length}</p>
                </div>
                <div class="history-item-actions">
                    <button onclick="dentalAnalyzer.viewHistoryDetail(${index})" class="btn btn-sm btn-primary">
                        View Details
                    </button>
                </div>
            </div>
        `).join('');
    }

    getConfidenceClass(confidence) {
        const percent = confidence * 100;
        if (percent >= 90) return 'high';
        if (percent >= 70) return 'medium';
        return 'low';
    }

    showModal(content) {
        const modal = document.createElement('div');
        modal.className = 'modal-overlay';
        modal.innerHTML = content;
        document.body.appendChild(modal);

        // Fade in animation
        requestAnimationFrame(() => {
            modal.style.opacity = '1';
            modal.querySelector('.modal-content').style.transform = 'translateY(0)';
        });

        // Close handlers
        const closeModal = () => {
            modal.style.opacity = '0';
            modal.querySelector('.modal-content').style.transform = 'translateY(20px)';
            setTimeout(() => modal.remove(), 300);
        };

        modal.querySelector('.modal-close').onclick = closeModal;
        modal.onclick = (e) => {
            if (e.target === modal) closeModal();
        };
        
        // Escape key handler
        const escHandler = (e) => {
            if (e.key === 'Escape') {
                closeModal();
                document.removeEventListener('keydown', escHandler);
            }
        };
        document.addEventListener('keydown', escHandler);
    }

    updateStatisticsOnNewPrediction(result) {
        // Get statistics elements
        const todayAnalyses = document.getElementById('todayAnalyses');
        const accuracyRate = document.getElementById('accuracyRate');
        const avgProcessingTime = document.getElementById('avgProcessingTime');
        const totalAnalyses = document.getElementById('totalAnalyses');

        // Update today's analyses count
        const currentCount = parseInt(todayAnalyses.textContent || '0');
        this.animateCounter(todayAnalyses, currentCount + 1);

        // Update accuracy rate
        const newAccuracy = result.predictions[0][1] * 100;
        const currentAccuracy = parseFloat(accuracyRate.textContent || '0');
        const avgAccuracy = this.cases.length === 1 ? 
            newAccuracy : 
            (currentAccuracy * (this.cases.length - 1) + newAccuracy) / this.cases.length;
        
        this.animateCounter(accuracyRate, avgAccuracy, '%');
        this.updateAccuracyRing(avgAccuracy);

        // Update processing time
        const newTime = result.processingTime || 2.5; // Default if not provided
        const currentTime = parseFloat(avgProcessingTime.textContent || '0');
        const avgTime = this.cases.length === 1 ? 
            newTime : 
            (currentTime * (this.cases.length - 1) + newTime) / this.cases.length;
        
        this.animateCounter(avgProcessingTime, avgTime, 's');
        
        // Update total analyses
        const total = parseInt(localStorage.getItem('totalAnalyses') || '0') + 1;
        localStorage.setItem('totalAnalyses', total.toString());
        this.animateCounter(totalAnalyses, total);

        // Update trends
        this.updateTrends(result);
    }

    updateTrends(newResult) {
        // Calculate trends based on recent results
        const recentCases = this.cases.slice(-5); // Last 5 cases
        
        // Today's analysis trend
        const todayTrend = document.querySelector('#todayAnalyses').closest('.stat-item').querySelector('.stat-trend');
        const todayChange = ((recentCases.length / 5) * 100).toFixed(0);
        todayTrend.innerHTML = `
            <i class="fas fa-arrow-${todayChange >= 0 ? 'up' : 'down'}"></i>
            <span>${Math.abs(todayChange)}%</span>
        `;
        todayTrend.className = `stat-trend ${todayChange >= 0 ? 'positive' : 'negative'}`;

        // Accuracy trend
        const accuracyTrend = document.querySelector('#accuracyRate').closest('.stat-item').querySelector('.stat-progress');
        const currentAccuracy = newResult.predictions[0][1] * 100;
        accuracyTrend.innerHTML = `
            <div class="progress-ring">
                <circle r="20" cx="25" cy="25"></circle>
            </div>
        `;
        this.updateAccuracyRing(currentAccuracy);

        // Processing time chart
        const timeData = {
            labels: recentCases.map((_, i) => `Case ${i + 1}`),
            values: recentCases.map(c => c.processingTime || 2.5)
        };
        this.updateProcessingTimeChart(timeData);

        // Total analyses trend
        const totalTrend = document.querySelector('#totalAnalyses').closest('.stat-item').querySelector('.stat-trend');
        const totalChange = 5; // Fixed 5% increase for demonstration
        totalTrend.innerHTML = `
            <i class="fas fa-arrow-up"></i>
            <span>${totalChange}%</span>
        `;
    }

    // Add cleanup method
    cleanup() {
        // Remove event listeners
        this.eventListeners.forEach(({ element, event, handler }) => {
            element.removeEventListener(event, handler);
        });
        this.eventListeners.clear();

        // Cleanup Chart.js instance
        if (this.timeChart) {
            this.timeChart.destroy();
            this.timeChart = null;
        }

        MemoryManager.cleanup();
    }

    async fetchWithTimeout(resource, options = {}) {
        const { timeout = 5000 } = options;
        
        const controller = new AbortController();
        const id = setTimeout(() => controller.abort(), timeout);
        
        try {
            const response = await fetch(resource, {
                ...options,
                signal: controller.signal
            });
            clearTimeout(id);
            return response;
        } catch (error) {
            clearTimeout(id);
            if (error.name === 'AbortError') {
                throw new Error('Request timed out');
            }
            throw error;
        }
    }

    async optimizeImage(file) {
        return new Promise((resolve) => {
            const img = new Image();
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            img.onload = () => {
                // Max dimensions
                const MAX_WIDTH = 2048;
                const MAX_HEIGHT = 2048;

                let width = img.width;
                let height = img.height;

                if (width > MAX_WIDTH) {
                    height *= MAX_WIDTH / width;
                    width = MAX_WIDTH;
                }
                if (height > MAX_HEIGHT) {
                    width *= MAX_HEIGHT / height;
                    height = MAX_HEIGHT;
                }

                canvas.width = width;
                canvas.height = height;

                // Apply sharpening
                ctx.filter = 'contrast(1.1) sharpen(1)';
                ctx.drawImage(img, 0, 0, width, height);

                canvas.toBlob((blob) => {
                    resolve(new File([blob], file.name, {
                        type: 'image/jpeg',
                        lastModified: Date.now()
                    }));
                }, 'image/jpeg', 0.85);
            };

            img.src = URL.createObjectURL(file);
        });
    }
}

class AdvancedAnalysis {
    constructor(imageData) {
        this.imageData = imageData;
        this.annotations = [];
        this.measurements = [];
    }

    addAnnotation(x, y, text) {
        this.annotations.push({ x, y, text });
    }

    addMeasurement(start, end) {
        const distance = Math.sqrt(
            Math.pow(end.x - start.x, 2) + 
            Math.pow(end.y - start.y, 2)
        );
        this.measurements.push({ start, end, distance });
    }

    generateReport() {
        return {
            annotations: this.annotations,
            measurements: this.measurements,
            timestamp: new Date().toISOString()
        };
    }
}

class CacheManager {
    constructor() {
        this.dbName = 'dentalAnalyzerCache';
        this.storeName = 'analyses';
    }

    async init() {
        const request = indexedDB.open(this.dbName, 1);
        
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            db.createObjectStore(this.storeName, { keyPath: 'id' });
        };

        return new Promise((resolve, reject) => {
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async cacheAnalysis(id, data) {
        const db = await this.init();
        const tx = db.transaction(this.storeName, 'readwrite');
        const store = tx.objectStore(this.storeName);
        await store.put({ id, data, timestamp: Date.now() });
    }

    async getCachedAnalysis(id) {
        const db = await this.init();
        const tx = db.transaction(this.storeName, 'readonly');
        const store = tx.objectStore(this.storeName);
        return store.get(id);
    }
}

class PerformanceMonitor {
    static startMeasure(label) {
        performance.mark(`${label}-start`);
    }

    static endMeasure(label) {
        performance.mark(`${label}-end`);
        performance.measure(label, `${label}-start`, `${label}-end`);
        
        const measure = performance.getEntriesByName(label)[0];
        console.debug(`${label}: ${measure.duration.toFixed(2)}ms`);
        
        return measure.duration;
    }
}

class MemoryManager {
    static cleanup() {
        // Clear object URLs
        const urls = Array.from(document.querySelectorAll('img[src^="blob:"]'))
            .map(img => img.src);
        urls.forEach(URL.revokeObjectURL);

        // Clear canvases
        const canvases = document.querySelectorAll('canvas');
        canvases.forEach(canvas => {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        });

        // Force garbage collection if possible
        if (window.gc) window.gc();
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new DentalAnalyzer();
});

// Update the toggleTheme function
function toggleTheme() {
    const body = document.body;
    const isDark = body.getAttribute('data-theme') === 'dark';
    const themeButton = document.querySelector('.side-button i');
    
    // Toggle theme
    body.setAttribute('data-theme', isDark ? 'light' : 'dark');
    
    // Update icon (moon for light theme, sun for dark theme)
    themeButton.className = isDark ? 'fas fa-moon' : 'fas fa-sun';
    
    // Save preference
    localStorage.setItem('theme', isDark ? 'light' : 'dark');
}

// Initialize theme on page load
document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    const themeButton = document.querySelector('.side-button i');
    
    document.body.setAttribute('data-theme', savedTheme);
    // Set initial icon (moon for light theme, sun for dark theme)
    themeButton.className = savedTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
});