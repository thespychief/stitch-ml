const cliProgress = require('cli-progress');

class ProgressBar {
  constructor(steps) {
    // this.steps;
    this.progressBar = new cliProgress.SingleBar(
      {}, cliProgress.Presets.shades_classic,
    );
    this.progressBar.start(steps, 0);
  }

  async update(step) {
    this.progressBar.update(step);
  }

  finish() {
    this.progressBar.update(this.steps + 1);
    this.progressBar.stop();
  }
}

module.exports = ProgressBar;
