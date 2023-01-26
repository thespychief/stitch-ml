class Matrix {
  static random(m, n) {
    const result = [];
    for (let i = 0; i < m; i++) {
      result[i] = [];
      for (let j = 0; j < n; j++) {
        result[i][j] = (Math.random() * 2) - 1;
      }
    }
    return result;
  }

  static zeros(m, n) {
    const result = [];
    for (let i = 0; i < m; i++) {
      result[i] = [];
      for (let j = 0; j < n; j++) {
        result[i][j] = 0;
      }
    }
    return result;
  }

  static add(m1, m2) {
    const result = [];
    for (let i = 0; i < m1.length; i++) {
      result[i] = [];
      for (let j = 0; j < m1[i].length; j++) {
        result[i][j] = m1[i][j] + m2[i][j];
      }
    }
    return result;
  }

  static subtract(m1, m2) {
    const result = [];
    for (let i = 0; i < m1.length; i++) {
      result[i] = [];
      for (let j = 0; j < m1[i].length; j++) {
        result[i][j] = m1[i][j] - m2[i][j];
      }
    }
    return result;
  }

  static hadamardProduct(m1, m2) {
    const result = [];
    for (let i = 0; i < m1.length; i++) {
      result[i] = [];
      for (let j = 0; j < m1[i].length; j++) {
        result[i][j] = m1[i][j] * m2[i][j];
      }
    }
    return result;
  }

  static product(m1, m2) {
    const result = [];
    for (let i = 0; i < m1.length; i++) {
      result[i] = [];
      for (let j = 0; j < m2[0].length; j++) {
        let sum = 0;
        for (let k = 0; k < m1[0].length; k++) {
          sum += m1[i][k] * m2[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  }

  static transpose(m) {
    const result = [];
    for (let i = 0; i < m[0].length; i++) {
      result[i] = Array(m.length);
    }
    for (let i = 0; i < m.length; i++) {
      for (let j = 0; j < m[0].length; j++) {
        result[j][i] = m[i][j];
      }
    }
    return result;
  }

  static map(m, func) {
    const result = m;
    for (let i = 0; i < m.length; i++) {
      for (let j = 0; j < m[i].length; j++) {
        const val = m[i][j];
        result[i][j] = func(val);
      }
    }
    return result;
  }

  static sample(m, size) {
    const sx = Math.floor(Math.random() * (m[0].length - size + 1));
    const sy = Math.floor(Math.random() * (m.length - size + 1));
    return m.slice(sy, sy + size).map((x) => x.slice(sx, sx + size));
  }

  /**
   * Not quite complete
   */
  static convolve(m, kernelSize) {
    const result = [];
    for (let i = 0; i < m.length - kernelSize + 1; i += kernelSize) {
      for (let j = 0; j < m[0].length - kernelSize + 1; j += kernelSize) {
        const ex = i + kernelSize;
        const ey = j + kernelSize;
        const section = m.slice(i, ex).map((x) => x.slice(j, ey));
        result.push(section);
      }
    }
    return result;
  }
}

module.exports = Matrix;
