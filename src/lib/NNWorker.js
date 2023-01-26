const { expose } = require('threads');
const Network = require('../Network');

expose(async (networkParams, trainingParams, trainingData) => {
  const network = new Network(networkParams);
  network.train({ trainingData, ...trainingParams });
  return network.save();
});
