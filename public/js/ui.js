class NeuralNetworkUI {
    constructor() {
        this.canvas = document.getElementById('neuralCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.points = [];
        this.network = new NeuralNetwork(2, 4, 1);
        this.network.setActivationFunction('relu');
        this.isTraining = false;
        this.setupEventListeners();
        this.resizeCanvas();
        this.draw();
        this.logEvent('Aplicação inicializada');
    }

    setupEventListeners() {
        // Canvas click handler
        this.canvas.addEventListener('click', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            this.addPoint(x, y);
        });

        // Training button
        document.getElementById('trainButton').addEventListener('click', () => {
            if (this.points.length > 0) {
                this.logEvent(`Iniciando treinamento com ${this.points.length} pontos`);
                this.trainNetwork();
            } else {
                this.logEvent('Nenhum ponto para treinar. Adicione pontos primeiro.');
            }
        });

        // Clear button
        document.getElementById('clearButton').addEventListener('click', () => {
            this.points = [];
            this.logEvent('Todos os pontos foram removidos');
            this.draw();
        });

        // Reset button
        document.getElementById('resetButton').addEventListener('click', () => {
            this.network = new NeuralNetwork(2, 4, 1);
            this.network.setActivationFunction(this.network.activationFunction);
            this.logEvent('Rede neural reinicializada');
            this.draw();
        });

        // Activation function selector
        document.getElementById('activationFunction').addEventListener('change', (e) => {
            this.network.setActivationFunction(e.target.value);
            this.logEvent(`Função de ativação alterada para: ${e.target.value}`);
            if (this.points.length > 0) {
                this.trainNetwork();
            }
        });

        // Learning rate slider
        const learningRateSlider = document.getElementById('learningRate');
        const learningRateValue = document.getElementById('learningRateValue');
        learningRateSlider.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            learningRateValue.textContent = value.toFixed(3);
            this.network.learningRate = value;
            this.logEvent(`Taxa de aprendizado ajustada para: ${value.toFixed(3)}`);
        });

        // Epochs slider
        const epochsSlider = document.getElementById('epochs');
        const epochsValue = document.getElementById('epochsValue');
        epochsSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            epochsValue.textContent = value;
            this.logEvent(`Número de épocas ajustado para: ${value}`);
        });

        // Window resize handler
        window.addEventListener('resize', () => {
            this.resizeCanvas();
            this.draw();
        });

        // Help modal
        const helpButton = document.getElementById('helpButton');
        const helpModal = document.getElementById('helpModal');
        const closeButton = helpModal.querySelector('.close-button');

        helpButton.addEventListener('click', () => {
            helpModal.classList.add('show');
            this.logEvent('Modal de ajuda aberto');
        });

        closeButton.addEventListener('click', () => {
            helpModal.classList.remove('show');
            this.logEvent('Modal de ajuda fechado');
        });

        // Close modal when clicking outside
        window.addEventListener('click', (e) => {
            if (e.target === helpModal) {
                helpModal.classList.remove('show');
                this.logEvent('Modal de ajuda fechado');
            }
        });

        // Random data button
        document.getElementById('randomDataButton').addEventListener('click', () => {
            this.generateRandomData();
        });

        // Random points slider
        const randomPointsSlider = document.getElementById('randomPoints');
        const randomPointsValue = document.getElementById('randomPointsValue');
        randomPointsSlider.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            randomPointsValue.textContent = value;
        });
    }

    logEvent(message) {
        const logBox = document.getElementById('logBox');
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = `[${timestamp}] ${message}\n`;
        logBox.textContent += logEntry;
        logBox.scrollTop = logBox.scrollHeight;
        console.log(message);
    }

    resizeCanvas() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
    }

    addPoint(x, y) {
        // Convert canvas coordinates to normalized coordinates (-1 to 1)
        const normalizedX = (x / this.canvas.width) * 2 - 1;
        const normalizedY = 1 - (y / this.canvas.height) * 2; // Flip Y axis

        // Add point with random class (0 or 1)
        const pointClass = Math.random() > 0.5 ? 1 : 0;
        this.points.push({
            x: normalizedX,
            y: normalizedY,
            class: pointClass
        });

        this.logEvent(`Ponto adicionado: (${normalizedX.toFixed(2)}, ${normalizedY.toFixed(2)}) - Classe ${pointClass}`);
        this.draw();
    }

    async trainNetwork() {
        if (this.isTraining) return;
        this.isTraining = true;

        const inputs = this.points.map(p => [p.x, p.y]);
        const targets = this.points.map(p => [p.class]);
        const epochs = parseInt(document.getElementById('epochs').value);

        this.logEvent(`Iniciando treinamento com ${epochs} épocas...`);

        // Train for specified epochs
        for (let epoch = 0; epoch < epochs; epoch++) {
            this.network.train(inputs, targets, 1);
            this.draw();
            await new Promise(resolve => setTimeout(resolve, 10));
        }

        this.isTraining = false;
        this.logEvent('Treinamento concluído');
    }

    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw decision boundary
        this.drawDecisionBoundary();

        // Draw points
        this.points.forEach(point => {
            this.ctx.beginPath();
            this.ctx.arc(
                (point.x + 1) * this.canvas.width / 2,
                (1 - point.y) * this.canvas.height / 2,
                5,
                0,
                Math.PI * 2
            );
            this.ctx.fillStyle = point.class === 1 ? '#4a6fa5' : '#dc3545';
            this.ctx.fill();
        });

        // Draw decision line
        this.drawDecisionLine();
    }

    drawDecisionBoundary() {
        const resolution = 20;
        const cellWidth = this.canvas.width / resolution;
        const cellHeight = this.canvas.height / resolution;

        for (let i = 0; i < resolution; i++) {
            for (let j = 0; j < resolution; j++) {
                const x = (i / resolution) * 2 - 1;
                const y = 1 - (j / resolution) * 2;
                const prediction = this.network.predict([x, y])[0];
                
                // Create a gradient effect
                const gradient = this.ctx.createLinearGradient(0, 0, 0, cellHeight);
                gradient.addColorStop(0, `rgba(74, 111, 165, ${prediction * 0.5})`);
                gradient.addColorStop(1, `rgba(74, 111, 165, ${prediction * 0.2})`);
                
                this.ctx.fillStyle = gradient;
                this.ctx.fillRect(
                    i * cellWidth,
                    j * cellHeight,
                    cellWidth,
                    cellHeight
                );
            }
        }
    }

    drawDecisionLine() {
        // Get the weights from the first layer
        const weights = this.network.weights1;
        const bias = this.network.bias1;

        // Calculate the decision boundary line
        const w1 = weights[0][0];
        const w2 = weights[1][0];
        const b = bias[0];

        // Calculate line endpoints
        const x1 = -1;
        const y1 = (-w1 * x1 - b) / w2;
        const x2 = 1;
        const y2 = (-w1 * x2 - b) / w2;

        // Convert to canvas coordinates
        const canvasX1 = (x1 + 1) * this.canvas.width / 2;
        const canvasY1 = (1 - y1) * this.canvas.height / 2;
        const canvasX2 = (x2 + 1) * this.canvas.width / 2;
        const canvasY2 = (1 - y2) * this.canvas.height / 2;

        // Draw the line
        this.ctx.beginPath();
        this.ctx.moveTo(canvasX1, canvasY1);
        this.ctx.lineTo(canvasX2, canvasY2);
        this.ctx.strokeStyle = 'rgba(0, 0, 0, 0.5)';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
    }

    generateRandomData() {
        const numPoints = parseInt(document.getElementById('randomPoints').value);
        this.points = [];
        
        for (let i = 0; i < numPoints; i++) {
            const x = Math.random() * 2 - 1; // -1 to 1
            const y = Math.random() * 2 - 1; // -1 to 1
            const pointClass = Math.random() > 0.5 ? 1 : 0;
            
            this.points.push({
                x: x,
                y: y,
                class: pointClass
            });
        }

        this.logEvent(`Gerados ${numPoints} pontos aleatórios`);
        this.draw();
    }
}

// Initialize the UI when the page loads
window.addEventListener('load', () => {
    new NeuralNetworkUI();
}); 