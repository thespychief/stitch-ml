/* eslint-disable no-undef */
const lodash = require('lodash');
const StitchML = require('../src/index');

test('train and evaluate from object', async () => {
  const xor = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [0] },
  ];

  const network = new StitchML.Network({
    structure: [2, 4, 1],
  });

  await network.train({ data: xor, epochs: 50000 });

  const results = await network.evaluate({
    data: xor,
    func: ({ output, prediction }) => (Math.round(prediction[0]) === output[0]),
  });

  expect(results.correct).toEqual(4);
  expect(results.incorrect).toEqual(0);
  expect(results.accuracyPrct).toEqual(100);
});

test('train and evaluate from json file', async () => {
  const network = new StitchML.Network({
    structure: [2, 4, 1],
  });

  await network.train({ data: './test/data/xor.json', epochs: 50000 });

  const results = await network.evaluate({
    data: './test/data/xor.json',
    func: ({ output, prediction }) => (Math.round(prediction[0]) === output[0]),
  });

  expect(results.correct).toEqual(4);
  expect(results.incorrect).toEqual(0);
  expect(results.accuracyPrct).toEqual(100);
});

jest.setTimeout(10000);
test('train and evaluate from ndjson file', async () => {
  const network = new StitchML.Network({
    structure: [2, 4, 1],
  });

  await network.train({ data: './test/data/xor.ndjson', epochs: 50000 });

  const results = await network.evaluate({
    data: './test/data/xor.ndjson',
    func: ({ output, prediction }) => (Math.round(prediction[0]) === output[0]),
  });

  expect(results.correct).toEqual(4);
  expect(results.incorrect).toEqual(0);
  expect(results.accuracyPrct).toEqual(100);
});

test('train, save, load, and evaluate', async () => {
  const xor = [
    { input: [0, 0], output: [0] },
    { input: [0, 1], output: [1] },
    { input: [1, 0], output: [1] },
    { input: [1, 1], output: [0] },
  ];

  const network = new StitchML.Network({
    structure: [2, 4, 1],
  });

  await network.train({ data: xor, epochs: 50000 });

  network.saveToFile({ file: './test/data/xor.net.json' });

  const networkFromFile = new StitchML.Network(
    StitchML.FileOps.loadModel({ file: './test/data/xor.net.json' }),
  );

  const results = await networkFromFile.evaluate({
    data: xor,
    func: ({ output, prediction }) => (Math.round(prediction[0]) === output[0]),
  });

  expect(results.correct).toEqual(4);
  expect(results.incorrect).toEqual(0);
  expect(results.accuracyPrct).toEqual(100);
});

jest.setTimeout(300000);
test('train and evaluate mnist', async () => {
  const network = new StitchML.Network({
    structure: [784, 100, 10],
  });

  await network.train({
    data: './test/data/mnist_train.json',
    showProgress: true,
  });

  const results = await network.evaluate({
    data: './test/data/mnist_test.json',
    func: ({ output, prediction }) => lodash.isEqual(
      lodash.indexOf(prediction, lodash.max(prediction)),
      lodash.indexOf(output, lodash.max(output)),
    ),
    showProgress: true,
  });

  expect(results.correct).toBeGreaterThan(9000);
  expect(results.incorrect).toBeLessThan(1000);
  expect(results.accuracyPrct).toBeGreaterThan(90);
});
