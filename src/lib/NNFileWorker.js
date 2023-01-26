const { expose } = require('threads');
const Network = require('../Network');

expose(async (networkParams, trainingParams, file) => {
  const network = new Network(networkParams);
  await network.trainFromFileStream({ filename: file, ...trainingParams });
  return network.save();
});
