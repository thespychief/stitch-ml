const { spawn, Pool, Worker } = require('threads');

const trainInParallel = async (networks) => {
  const pool = Pool(() => spawn(new Worker('./lib/NNWorker')));

  const results = [];
  for (let i = 0; i < networks.length; i++) {
    pool.queue(async (nnWorker) => {
      const result = await nnWorker(
        networks[i].network, networks[i].training, networks[i].data,
      );
      results.push(result);
    });
  }

  await pool.completed();
  await pool.terminate();

  return results;
};

const trainInParallelFromFiles = async (networks) => {
  const pool = Pool(() => spawn(new Worker('./lib/NNFileWorker')));

  const results = [];
  for (let i = 0; i < networks.length; i++) {
    pool.queue(async (nnFileWorker) => {
      const result = await nnFileWorker(
        networks[i].network, networks[i].training, networks[i].file,
      );
      results.push(result);
    });
  }

  await pool.completed();
  await pool.terminate();

  return results;
};

module.exports = {
  trainInParallel,
  trainInParallelFromFiles,
};
