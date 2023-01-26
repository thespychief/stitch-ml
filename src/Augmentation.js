const _ = require('lodash');

module.exports = {
  JitterPointsInGrid: (grid, shift, cutoff) => {
    const randomIntInRangeInclusive = (range) => Math.floor(
      Math.random() * (range * 2 + 1),
    ) - range;

    const result = _.cloneDeep(grid);
    for (let y = 0; y < grid.length; y++) {
      for (let x = 0; x < grid[0].length; x++) {
        if (grid[x][y] === cutoff) {
          const xShift = randomIntInRangeInclusive(shift);
          const yShift = randomIntInRangeInclusive(shift);

          let newX = x + xShift;
          let newY = y + yShift;

          if (newX < 0) newX = 0;
          if (newX > grid.length - 1) newX = grid.length - 1;

          if (newY < 0) newY = 0;
          if (newY > grid.length - 1) newY = grid.length - 1;

          result[x][y] = 0;
          result[newX][newY] = 1;
        }
      }
    }
    return result;
  },
};
