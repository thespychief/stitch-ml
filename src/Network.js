/* eslint-disable no-await-in-loop */
/* eslint-disable no-restricted-syntax */
const fs = require('fs');
const Readline = require('readline');
const EventEmitter = require('events');
const _ = require('lodash');

const Functions = require('./Activation');
const FileOps = require('./FileOps');
const Matrix = require('./lib/Matrix');
const Segue = require('./lib/Segue');

const ProgressBar = require('./ProgressBar');

class Network {
  constructor({
    id = 0,
    structure,
    segues,
    activation = 'Sigmoid',
    useZeros = false,
    history = [],
  }) {
    this.id = id;
    this.structure = structure;
    this.segues = [];
    this.layerCount = structure.length - 1;
    this.activation = activation;
    this.history = history;

    for (let i = 0; i < this.layerCount; i++) {
      this.segues[i] = new Segue(structure[i], structure[i + 1], useZeros);
    }

    if (segues) {
      for (let i = 0; i < this.segues.length; i++) {
        this.segues[i].updateWeights(segues[i].weights);
        this.segues[i].updateBias(segues[i].bias);
      }
    }

    this.eventEmitter = new EventEmitter();
  }

  async train({
    data,
    epochs = 1,
    learningRate = 0.1,
    showProgress = false,
  }) {
    if (_.isArray(data)) {
      this.trainFromData({
        data, epochs, learningRate, showProgress,
      });
    } else if (_.isString(data)) {
      const fileType = data.split('.').pop();
      if (fileType === 'json') {
        const dataFromFile = JSON.parse(fs.readFileSync(data));
        this.trainFromData({
          data: dataFromFile, epochs, learningRate, showProgress,
        });
      } else if (fileType === 'ndjson') {
        await this.trainFromFileStream({
          file: data, epochs, learningRate, showProgress,
        });
      }
    }
  }

  trainFromData({
    data,
    epochs = 1,
    learningRate = 0.1,
    dotProductFunc = Matrix.product,
    showProgress = false,
  }) {
    const progress = showProgress
      ? new ProgressBar(epochs * data.length) : null;

    const startTime = Date.now();
    for (let j = 0; j < epochs; j++) {
      const shuffledData = _.shuffle(data);

      for (let i = 0; i < shuffledData.length; i++) {
        if (i % 1000 === 0) {
          this.eventEmitter.emit('event', {
            iteration: i,
            total: shuffledData.length,
          });
        }

        if (showProgress) progress.update((data.length * j) + i + 1);

        this.backprop(
          shuffledData[i].input,
          shuffledData[i].output,
          learningRate,
          dotProductFunc,
        );
      }

      this.eventEmitter.emit('event', {
        epoch: j + 1,
        iteration: shuffledData.length,
        total: shuffledData.length,
      });
    }
    const endTime = Date.now();

    this.history.push({
      event: 'train',
      epochs,
      learningRate,
      time: `${((endTime - startTime) / 1000 / 60).toFixed(2)}m`,
    });
  }

  async trainFromFileStream({
    file,
    epochs = 1,
    learningRate = 0.1,
    dotProductFunc = Matrix.product,
    showProgress = false,
  }) {
    const lineCount = await FileOps.getLineCountInFile({ file });
    const progress = showProgress
      ? new ProgressBar(epochs * lineCount) : null;

    const startTime = Date.now();
    for (let j = 0; j < epochs; j++) {
      const fileStream = fs.createReadStream(file);
      const rl = Readline.createInterface({
        input: fileStream,
      });

      let i = 0;
      for await (const line of rl) {
        // validation
        if (line.trim() !== '') {
          const point = JSON.parse(line);
          this.backprop(
            point.input, point.output, learningRate, dotProductFunc,
          );
        }

        if (showProgress) progress.update((lineCount * j) + i + 1);
        i += 1;
      }
    }
    const endTime = Date.now();

    this.history.push({
      event: 'train',
      epochs,
      learningRate,
      time: `${((endTime - startTime) / 1000 / 60).toFixed(2)}m`,
    });
  }

  backprop(inputArray, targetArray, learningRate, dotProduct) {
    const input = _.chunk(inputArray);
    const layerResult = [];
    layerResult[0] = input;
    for (let i = 0; i < this.layerCount; i++) {
      layerResult[i + 1] = dotProduct(
        this.segues[i].getWeights(),
        layerResult[i],
      );
      layerResult[i + 1] = Matrix.add(
        layerResult[i + 1],
        this.segues[i].getBias(),
      );
      layerResult[i + 1] = Matrix.map(
        layerResult[i + 1],
        Functions[this.activation].equation,
      );
    }

    const targets = _.chunk(targetArray);
    const layerErrors = [];
    layerErrors[this.layerCount] = Matrix.subtract(
      targets, layerResult[this.layerCount],
    );

    const gradients = [];
    for (let i = this.layerCount; i > 0; i--) {
      gradients[i] = Matrix.map(
        layerResult[i], Functions[this.activation].derivative,
      );
      gradients[i] = Matrix.hadamardProduct(gradients[i], layerErrors[i]);
      gradients[i] = Matrix.map(gradients[i], (x) => x * learningRate);

      const hiddenTranspose = Matrix.transpose(layerResult[i - 1]);
      const weightDeltas = dotProduct(gradients[i], hiddenTranspose);

      this.segues[i - 1].addToWeights(weightDeltas);
      this.segues[i - 1].addToBias(gradients[i]);

      layerErrors[i - 1] = dotProduct(
        Matrix.transpose(this.segues[i - 1].getWeights()), layerErrors[i],
      );
    }
  }

  predictWithConvolution(input, kernelSize, filterFunc) {
    const predictions = [];
    for (let i = 0; i < input.length - kernelSize + 1; i++) {
      for (let j = 0; j < input[0].length - kernelSize + 1; j++) {
        const ex = i + kernelSize;
        const ey = j + kernelSize;
        const section = input.slice(i, ex).map((x) => x.slice(j, ey));

        const pixels = _.flatten(section);
        const prediction = this.predict(pixels);

        if (filterFunc(prediction)) {
          predictions.push({
            x1: i,
            y1: j,
            x2: ex,
            y2: ey,
            prediction,
          });
        }
      }
    }
    return predictions;
  }

  /**
   * TODO: Modify to cover left and bottom edges
   */
  predictWithSparseConvolution(input, kernelSize, paddingFactor, filterFunc) {
    const padding = Math.floor(kernelSize / paddingFactor);
    const predictions = [];
    for (let i = 0; i < input.length - kernelSize + 1; i += padding) {
      for (let j = 0; j < input[0].length - kernelSize + 1; j += padding) {
        const ex = i + kernelSize;
        const ey = j + kernelSize;
        const section = input.slice(i, ex).map((x) => x.slice(j, ey));

        const pixels = _.flatten(section);
        const prediction = this.predict(pixels);

        if (filterFunc(prediction)) {
          predictions.push({
            x1: i,
            y1: j,
            x2: ex,
            y2: ey,
            prediction,
          });
        }
      }
    }
    return predictions;
  }

  predict(input) {
    let layerResult = _.chunk(input);
    for (let i = 0; i < this.layerCount; i++) {
      layerResult = Matrix.product(this.segues[i].getWeights(), layerResult);
      layerResult = Matrix.add(this.segues[i].getBias(), layerResult);
      layerResult = Matrix.map(
        layerResult, Functions[this.activation].equation,
      );
    }

    return _.flatten(layerResult);
  }

  // eslint-disable-next-line consistent-return
  async evaluate({
    data,
    func,
    showProgress = false,
  }) {
    if (_.isArray(data)) {
      return this.evaluateFromData({
        data, func, showProgress,
      });
    }

    if (_.isString(data)) {
      const fileType = data.split('.').pop();
      if (fileType === 'json') {
        const dataFromFile = JSON.parse(fs.readFileSync(data));
        return this.evaluateFromData({
          data: dataFromFile, func, showProgress,
        });
      }

      if (fileType === 'ndjson') {
        return this.evaluateFromFileStream({
          file: data, func, showProgress,
        });
      }
    }
  }

  evaluateFromData({
    data,
    func,
    showProgress,
  }) {
    const progress = showProgress
      ? new ProgressBar(data.length) : null;

    let numCorrect = 0;
    let numIncorrect = 0;
    for (let i = 0; i < data.length; i++) {
      const point = data[i];
      const prediction = this.predict(point.input);
      const isCorrect = func({ output: point.output, prediction });
      // eslint-disable-next-line no-unused-expressions, no-plusplus
      isCorrect ? numCorrect++ : numIncorrect++;

      if (showProgress) progress.update(i + 1);
    }

    return {
      correct: numCorrect,
      incorrect: numIncorrect,
      accuracyPrct: (numCorrect / data.length) * 100,
    };
  }

  async evaluateFromFileStream({
    file,
    func,
    showProgress = false,
  }) {
    const lineCount = await FileOps.getLineCountInFile({ file });
    const progress = showProgress
      ? new ProgressBar(lineCount) : null;

    const fileStream = fs.createReadStream(file);
    const rl = Readline.createInterface({
      input: fileStream,
    });

    let numCorrect = 0;
    let numIncorrect = 0;
    let i = 0;
    for await (const line of rl) {
      if (line.trim() !== '') {
        const point = JSON.parse(line);
        const prediction = this.predict(point.input);
        const isCorrect = func({ output: point.output, prediction });
        // eslint-disable-next-line no-unused-expressions, no-plusplus
        isCorrect ? numCorrect++ : numIncorrect++;
      }
      if (showProgress) progress.update(i + 1);
      i += 1;
    }

    return {
      correct: numCorrect,
      incorrect: numIncorrect,
      accuracyPrct: (numCorrect / (numCorrect + numIncorrect)) * 100,
    };
  }

  saveToFile({ file }) {
    fs.writeFileSync(file, JSON.stringify({
      ...this,
    }));
  }
}

module.exports = Network;
