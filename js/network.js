class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize, learningRate = 0.01) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        this.learningRate = learningRate;

        // Initialize weights with random values
        this.weights1 = this.initializeWeights(inputSize, hiddenSize);
        this.weights2 = this.initializeWeights(hiddenSize, outputSize);
        
        // Initialize biases
        this.bias1 = new Array(hiddenSize).fill(0);
        this.bias2 = new Array(outputSize).fill(0);
    }

    initializeWeights(rows, cols) {
        return Array.from({ length: rows }, () =>
            Array.from({ length: cols }, () => Math.random() * 2 - 1)
        );
    }

    setActivationFunction(func) {
        this.activationFunction = func;
    }

    relu(x) {
        return Math.max(0, x);
    }

    reluDerivative(x) {
        return x > 0 ? 1 : 0;
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    sigmoidDerivative(x) {
        const sig = this.sigmoid(x);
        return sig * (1 - sig);
    }

    activate(x) {
        if (this.activationFunction === 'relu') {
            return this.relu(x);
        } else {
            return this.sigmoid(x);
        }
    }

    activateDerivative(x) {
        if (this.activationFunction === 'relu') {
            return this.reluDerivative(x);
        } else {
            return this.sigmoidDerivative(x);
        }
    }

    forward(input) {
        // First layer
        this.hidden = new Array(this.hiddenSize).fill(0);
        for (let i = 0; i < this.hiddenSize; i++) {
            let sum = this.bias1[i];
            for (let j = 0; j < this.inputSize; j++) {
                sum += input[j] * this.weights1[j][i];
            }
            this.hidden[i] = this.activate(sum);
        }

        // Output layer
        this.output = new Array(this.outputSize).fill(0);
        for (let i = 0; i < this.outputSize; i++) {
            let sum = this.bias2[i];
            for (let j = 0; j < this.hiddenSize; j++) {
                sum += this.hidden[j] * this.weights2[j][i];
            }
            this.output[i] = this.activate(sum);
        }

        return this.output;
    }

    backward(input, target) {
        // Calculate output layer error
        const outputError = this.output.map((o, i) => target[i] - o);
        const outputDelta = outputError.map((e, i) => e * this.activateDerivative(this.output[i]));

        // Calculate hidden layer error
        const hiddenError = new Array(this.hiddenSize).fill(0);
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.outputSize; j++) {
                hiddenError[i] += outputDelta[j] * this.weights2[i][j];
            }
        }
        const hiddenDelta = hiddenError.map((e, i) => e * this.activateDerivative(this.hidden[i]));

        // Update weights and biases
        for (let i = 0; i < this.hiddenSize; i++) {
            for (let j = 0; j < this.outputSize; j++) {
                this.weights2[i][j] += this.learningRate * outputDelta[j] * this.hidden[i];
            }
            this.bias2[i] += this.learningRate * outputDelta[i];
        }

        for (let i = 0; i < this.inputSize; i++) {
            for (let j = 0; j < this.hiddenSize; j++) {
                this.weights1[i][j] += this.learningRate * hiddenDelta[j] * input[i];
            }
        }

        for (let i = 0; i < this.hiddenSize; i++) {
            this.bias1[i] += this.learningRate * hiddenDelta[i];
        }
    }

    train(inputs, targets, epochs = 1000) {
        for (let epoch = 0; epoch < epochs; epoch++) {
            for (let i = 0; i < inputs.length; i++) {
                this.forward(inputs[i]);
                this.backward(inputs[i], targets[i]);
            }
        }
    }

    predict(input) {
        return this.forward(input);
    }
} 