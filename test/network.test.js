/* eslint-disable no-undef */
const fs = require('fs');
const _ = require('lodash');
const Network = require('../src/Network');

const trainingData = JSON.parse(
  fs.readFileSync('./test/data/mnist_train.json'),
);

const testData = JSON.parse(
  fs.readFileSync('./test/data/mnist_test.json'),
);

test('train and save', () => {
  const structure = [784, 100, 10];
  const network = new Network({
    structure,
  });

  network.train({ data: trainingData, epochs: 1 });
  const trainedNetwork = network.save();

  fs.writeFileSync('./test/data/mnist_network.json', JSON.stringify(
    trainedNetwork,
  ));

  expect(trainedNetwork.structure).toEqual(structure);
});

test('load and evaluate', () => {
  const model = JSON.parse(fs.readFileSync('./test/data/mnist_network.json'));
  const network = new Network(model);

  const results = [];
  for (let i = 0; i < testData.length; i++) {
    const point = testData[i];
    const prediction = network.predict(point.input);
    results.push([
      _.indexOf(prediction, _.max(prediction)),
      _.indexOf(point.output, _.max(point.output)),
    ]);
  }

  const matches = results.filter((arr) => _.isEqual(arr[0], arr[1]));

  const numMatches = matches.length;
  const accuracy = matches.length / results.length;

  expect(numMatches).toBeGreaterThanOrEqual(9200);
  expect(numMatches).toBeLessThan(9800);

  expect(accuracy).toBeGreaterThanOrEqual(0.92);
  expect(accuracy).toBeLessThan(0.98);
});

jest.setTimeout(300000);
test('train from file and evaluate', async () => {
  const structure = [784, 100, 10];
  const network = new Network({
    structure,
  });

  await network.train({
    file: './test/data/mnist_train.ndjson', epochs: 1,
  });

  const results = [];
  for (let i = 0; i < testData.length; i++) {
    const point = testData[i];
    const prediction = network.predict(point.input);
    results.push([
      _.indexOf(prediction, _.max(prediction)),
      _.indexOf(point.output, _.max(point.output)),
    ]);
  }

  const matches = results.filter((arr) => _.isEqual(arr[0], arr[1]));

  const numMatches = matches.length;
  const accuracy = matches.length / results.length;

  expect(numMatches).toBeGreaterThanOrEqual(9200);
  expect(numMatches).toBeLessThan(9800);

  expect(accuracy).toBeGreaterThanOrEqual(0.92);
  expect(accuracy).toBeLessThan(0.98);
});
