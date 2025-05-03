// Utility functions for neural network operations

/**
 * Normalizes a value from one range to another
 * @param {number} value - The value to normalize
 * @param {number} min - The minimum value of the input range
 * @param {number} max - The maximum value of the input range
 * @param {number} newMin - The minimum value of the output range
 * @param {number} newMax - The maximum value of the output range
 * @returns {number} The normalized value
 */
function normalize(value, min, max, newMin, newMax) {
    return ((value - min) / (max - min)) * (newMax - newMin) + newMin;
}

/**
 * Generates a random number between min and max
 * @param {number} min - The minimum value
 * @param {number} max - The maximum value
 * @returns {number} A random number between min and max
 */
function random(min, max) {
    return Math.random() * (max - min) + min;
}

/**
 * Calculates the mean squared error between two arrays
 * @param {Array} predictions - Array of predicted values
 * @param {Array} targets - Array of target values
 * @returns {number} The mean squared error
 */
function meanSquaredError(predictions, targets) {
    if (predictions.length !== targets.length) {
        throw new Error('Arrays must have the same length');
    }

    let sum = 0;
    for (let i = 0; i < predictions.length; i++) {
        sum += Math.pow(predictions[i] - targets[i], 2);
    }
    return sum / predictions.length;
}

/**
 * Creates a 2D array filled with zeros
 * @param {number} rows - Number of rows
 * @param {number} cols - Number of columns
 * @returns {Array} A 2D array filled with zeros
 */
function zeros(rows, cols) {
    return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

/**
 * Transposes a 2D array
 * @param {Array} matrix - The matrix to transpose
 * @returns {Array} The transposed matrix
 */
function transpose(matrix) {
    return matrix[0].map((_, i) => matrix.map(row => row[i]));
}

/**
 * Performs matrix multiplication
 * @param {Array} a - First matrix
 * @param {Array} b - Second matrix
 * @returns {Array} The resulting matrix
 */
function matrixMultiply(a, b) {
    const result = zeros(a.length, b[0].length);
    for (let i = 0; i < a.length; i++) {
        for (let j = 0; j < b[0].length; j++) {
            for (let k = 0; k < a[0].length; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return result;
}

/**
 * Performs element-wise addition of two matrices
 * @param {Array} a - First matrix
 * @param {Array} b - Second matrix
 * @returns {Array} The resulting matrix
 */
function matrixAdd(a, b) {
    return a.map((row, i) => row.map((val, j) => val + b[i][j]));
}

/**
 * Performs element-wise multiplication of two matrices
 * @param {Array} a - First matrix
 * @param {Array} b - Second matrix
 * @returns {Array} The resulting matrix
 */
function matrixMultiplyElementWise(a, b) {
    return a.map((row, i) => row.map((val, j) => val * b[i][j]));
}

// Export utility functions
window.utils = {
    normalize,
    random,
    meanSquaredError,
    zeros,
    transpose,
    matrixMultiply,
    matrixAdd,
    matrixMultiplyElementWise
}; 