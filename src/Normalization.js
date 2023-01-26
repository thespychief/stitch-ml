const _ = require('lodash');

module.exports = {
  NormalizeByConstant: (arr, c) => arr.map((x) => x / c),
  NormalizeByMax: (arr) => {
    const max = _.max(arr);
    return arr.map((x) => x / max);
  },
};
