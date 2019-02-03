const tf = require('@tensorflow/tfjs');

/**
 * Compute the euclidean distance between two vectors.
 * @param {tf.Tensor} a
 * @param {tf.Tensor} b
 */
export const euclidean = (a, b) => {
    return a.sub(b).norm('euclidean');
};

// Adapted from https://github.com/tensorflow/tfjs-examples/blob/master/polynomial-regression-core/index.js
export const curveFit = async (predict, xs, ys) => {
  
    // Step 1. Set up variables. Already done.
    
    // Step 2. Create an optimizer, we will use this later. You can play
    // with some of these values to see how the model performs.
    const numIterations = 75;
    const learningRate = 0.5;
    const optimizer = tf.train.sgd(learningRate);
    
    // Step 3. Write our training process functions.
    
    let loss = (prediction, labels) => {
      const error = prediction.sub(labels).square().mean();
      return error;
    };
    
    let train = async (xs, ys, numIterations) => {
      for (let i = 0; i < numIterations; i++) {
          // optimizer.minimize is where the training happens.
  
          // The function it takes must return a numerical estimate (i.e. loss)
          // of how well we are doing using the current state of
          // the variables we created at the start.
  
          // This optimizer does the 'backward' step of our training process
          // updating variables defined previously in order to minimize the
          // loss.
          optimizer.minimize(() => {
            // Feed the examples into the model
            const pred = predict(xs);
            return loss(pred, ys);
          });
  
          // Use tf.nextFrame to not block the browser.
          await tf.nextFrame();
      }
    };
    
    await train(xs, ys, numIterations);
};

/**
 * Fit a, b params for the differentiable curve used in lower dimensional fuzzy simplicial complex construction.
 * We want the smooth curve (from a pre-defined family with simple gradient) 
 * that best matches an offset exponential decay.
 */
export const findABParams = async (spread, minDist) => {
    const a = tf.variable(tf.scalar(Math.random()));
    const b = tf.variable(tf.scalar(Math.random()));
    const curve = (x) => {
      // 1.0 / (1.0 + a * x ** (2 * b))
      return tf.tidy(() => {
        return tf.scalar(1).div(
          tf.scalar(1).add(
            a.mul(
              x.pow(
                tf.scalar(2).mul(b)
              )
            )
          )
        );
      });
    };
    let xv = tf.linspace(0, spread * 3, 300);
    let yv = tf.zeros(xv.shape);
    // yv[xv < min_dist] = 1.0
    yv.where(xv.less(tf.scalar(minDist)), tf.ones(xv.shape));
    // yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    yv.where(
      xv.greaterEqual(tf.scalar(minDist)), 
      tf.exp(
        tf.scalar(0).sub(
          xv.sub(
            tf.scalar(minDist)
          ).div(
            tf.scalar(spread)
          )
        )
      )
    );
    await curveFit(curve, xv, yv);
    return { a, b };
};

  /**
 * @returns {array} An array of 3 ints representing the rng state.
 */
export const makeRngState = () => {
    const randInt = () => {
      return Math.floor(
        Math.random() * (Number.MAX_SAFE_INTEGER - Number.MIN_SAFE_INTEGER)
      ) + Number.MIN_SAFE_INTEGER;
    };
    return [randInt(), randInt(), randInt()];
};

/*
 * A fast (pseudo)-random number generator.
 * @param {array} state (3, ) The internal state of the rng.
 * @returns {int} A (pseudo)-random int value.
 */
export const tauRandInt = (state) => {
    state[0] = (((state[0] & 4294967294) << 12) & 0xffffffff) ^ (
          (((state[0] << 13) & 0xffffffff) ^ state[0]) >> 19
    );
    state[1] = (((state[1] & 4294967288) << 4) & 0xffffffff) ^ (
          (((state[1] << 2) & 0xffffffff) ^ state[1]) >> 25
    );
    state[2] = (((state[2] & 4294967280) << 17) & 0xffffffff) ^ (
          (((state[2] << 3) & 0xffffffff) ^ state[2]) >> 11
    );
    return state[0] ^ state[1] ^ state[2];
};

/*
 * A fast (pseudo)-random number generator for floats in the range [0, 1]
 * @param {array} state (3, ) The internal state of the rng.
 * @returns {float} A (pseudo)-random float in the interval [0, 1]
 */
export const tauRand = (state) => {
    let randInt = tauRandInt(state);
    return randInt / 0x7fffffff;
};