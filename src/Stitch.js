/* eslint-disable no-await-in-loop */
/* eslint-disable no-restricted-syntax */
const fs = require('fs');
const Readline = require('readline');
const _ = require('lodash');
const Matrix = require('./lib/Matrix');
const Network = require('./Network');

const { trainInParallelFromFiles } = require('./MetaLearning');

const combineOutputs = (outputs) => {
  const averages = [];
  for (let i = 0; i < outputs[0].length; i++) {
    let average = 0;
    for (let j = 0; j < outputs.length; j++) {
      average += outputs[j][i];
    }
    averages.push(average / outputs.length);
  }
  return averages;
};

class Stitch {
  constructor({
    id = 0,
    subnets = [],
  }) {
    this.id = id;

    if (subnets.length > 0) {
      this.subnets = subnets.map((subnet) => new Network(subnet));
    }
  }

  async create({
    data,
    subnets,
  }) {
    const readInterface = Readline.createInterface({
      input: fs.createReadStream(data.file),
    });

    for (let i = 0; i < subnets.count; i++) {
      fs.writeFileSync(`${data.tmpDir}/tmp/${i}.ndjson`, '');
    }

    for await (const line of readInterface) {
      const point = JSON.parse(line);
      const inputMatrix = _.chunk(point.input, data.dimension[1]);

      const convolutions = Matrix.convolve(inputMatrix, subnets.dimension[1]);
      for (let j = 0; j < convolutions.length; j++) {
        fs.appendFileSync(`${data.tmpDir}/tmp/${j}.ndjson`, `${JSON.stringify({
          input: _.flattenDeep(convolutions[j]),
          output: point.output,
        })}\n`);
      }
    }

    const networks = [];
    for (let i = 0; i < subnets.count; i++) {
      networks.push({
        network: {
          id: i, structure: subnets.structure, activation: subnets.activation,
        },
        training: {},
        file: `${data.tmpDir}/tmp/${i}.ndjson`,
      });
    }

    const trainedSubnets = await trainInParallelFromFiles(networks);

    const orderedSubnets = _.orderBy(trainedSubnets, ['id'], ['asc']);
    this.subnets = orderedSubnets.map((subnet) => new Network(subnet));
  }

  predict(input, subnetDimension) {
    const convolutions = Matrix.convolve(input, subnetDimension[1]);
    const outputs = [];
    for (let i = 0; i < this.subnets.length; i++) {
      outputs.push(this.subnets[i].predict(_.flattenDeep(convolutions[i])));
    }
    return outputs;
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

        const prediction = combineOutputs(this.predict(section, [16, 16]));

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

  save() {
    return {
      ...this,
    };
  }
}

module.exports = Stitch;
