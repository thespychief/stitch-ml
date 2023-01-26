const Matrix = require('./Matrix');

class Segue {
  constructor(prevLayerSize, layerSize, useZeros) {
    if (useZeros) {
      this.weights = Matrix.zeros(layerSize, prevLayerSize);
      this.bias = Matrix.zeros(layerSize, 1);
    } else {
      this.weights = Matrix.random(layerSize, prevLayerSize);
      this.bias = Matrix.random(layerSize, 1);
    }
  }

  getWeights() {
    return this.weights;
  }

  getBias() {
    return this.bias;
  }

  updateWeights(weights) {
    this.weights = weights;
  }

  updateBias(bias) {
    this.bias = bias;
  }

  addToWeights(deltaWeight) {
    this.weights = Matrix.add(this.weights, deltaWeight);
  }

  addToBias(bias) {
    this.bias = Matrix.add(this.bias, bias);
  }
}

module.exports = Segue;
