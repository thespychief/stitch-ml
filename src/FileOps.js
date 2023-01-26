const fs = require('fs');

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

module.exports = { Writer };
