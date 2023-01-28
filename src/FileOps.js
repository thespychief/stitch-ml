const fs = require('fs');
const Readline = require('readline');

const loadModel = ({ file }) => JSON.parse(fs.readFileSync(file));

const getLineCountInFile = async ({ file }) => {
  const readInterface = Readline.createInterface({
    input: fs.createReadStream(file),
  });

  let lineCount = 0;
  // eslint-disable-next-line no-restricted-syntax, no-unused-vars
  for await (const line of readInterface) {
    // eslint-disable-next-line no-plusplus
    lineCount++;
  }
  return lineCount;
};

class Writer {
  constructor({
    trainFile,
    testFile,
  }) {
    this.trainStream = fs.createWriteStream(
      trainFile, { flags: 'a' },
    );

    this.testStream = fs.createWriteStream(
      testFile, { flags: 'a' },
    );
  }

  writeLine(object, type) {
    const line = `${JSON.stringify(object)}\n`;
    if (type === 'train') {
      this.trainStream.write(line);
    } else if (type === 'test') {
      this.testStream.write(line);
    }
  }

  close() {
    this.trainStream.close();
    this.testStream.close();
  }
}

module.exports = { Writer, loadModel, getLineCountInFile };
