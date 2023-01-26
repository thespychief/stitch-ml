Stitch ML
========

> **This package is still in Alpha.** All future versions are subject to
breaking changes until the official first version, at which point the full
documentation will be released and Semantic Versioning will be used.

Stitch ML is an experimental machine learning library written in pure
Javascript.

## Overview

### Installation

```cmd
npm install stitch-ml --save
```

## Usage

With CommonJS:
```js
const StitchML = require('stitch-ml');
```

With ES Modules:
```js
import StitchML from 'stitch-ml';
```

### Input Data Format
```javascript
0 <= x <= 1

[
  {
    "input": [x, x, x, x, x, x, ...],
    "output": [x, x, x, x, ...]
  },
  ...
]
```

### Examples

```javascript
// Create a new network
const network = new StitchML.Network({
  structure: [784, 100, 10],
});

// Train the network for one epoch
network.train({ data: trainingData });

// Predict a single point
network.predict(input);

// Save the network
const modelToSave = network.save();
fs.writeFileSync('model.json', JSON.stringify(modelToSave));

// Load a network
const modelToLoad = JSON.parse(fs.readFileSync('model.json'));
const networkFromModel = new Stitch.Network(modelToLoad);
```

### Performance

##### MNIST

```
Dataset: MNIST Handwritten Digits
Device: 2014 Macbook Pro, 2.5 GHz Quad-Core Intel Core i7
Training Set Size: 60000, One Epoch
Test Set Size: 10000
---------------------------------------------------------
Training Time: 2:57.259 (m:ss.mmm)
Prediction Time: 0.789 ms
Accuracy: 9441 / 10000
```
