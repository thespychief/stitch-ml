const EventEmitter = require('events');

const Activation = require('./Activation');
const Augmentation = require('./Augmentation');
const Network = require('./Network');
const Stitch = require('./Stitch');
const Normalization = require('./Normalization');
const Matrix = require('./lib/Matrix');
const FileOps = require('./lib/Matrix');
const { trainInParallel, trainInParallelFromFiles } = require('./MetaLearning');

const Events = new EventEmitter();

module.exports = {
  Activation,
  Augmentation,
  Events,
  Network,
  Normalization,
  Matrix,
  Stitch,
  FileOps,
  trainInParallel,
  trainInParallelFromFiles,
};
