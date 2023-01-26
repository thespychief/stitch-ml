module.exports = {
  Sigmoid: {
    equation: (x) => (1 / (1 + Math.exp(-x))),
    derivative: (x) => (x * (1 - x)),
  },
  Tanh: {
    equation: (x) => Math.tanh(x),
    derivative: (x) => (1 - Math.tanh(x) ** 2),
  },
  ReLU: {
    equation: (x) => Math.max(0, x),
    derivative: (x) => (x <= 0 ? 0 : 1),
  },
  LeakyReLU: {
    equation: (x) => (x > 0 ? x : 0.01 * x),
    derivative: (x) => (x <= 0 ? 0.01 : 1),
  },
  Swish: {
    equation: (x) => x * (1 / (1 + Math.exp(-x))),
    derivative: (x) => (
      (1 / (1 + Math.exp(-x)))
      * (1 - (x * (1 / (1 + Math.exp(-x)))))
      + (x * (1 / (1 + Math.exp(-x))))
    ),
  },
};
