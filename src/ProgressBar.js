const cliProgress = require('cli-progress');

class ProgressBar {
  constructor(steps) {
    this.steps = steps;
    this.progressBar = new cliProgress.SingleBar(
      {}, cliProgress.Presets.shades_classic,
    );
    this.progressBar.start(steps, 0);
  }

  async update(step) {
    this.progressBar.update(step);
    if (this.steps === step) this.finish();
  }

  finish() {
    this.progressBar.update(this.steps);
    this.progressBar.stop();
  }
}

module.exports = ProgressBar;
