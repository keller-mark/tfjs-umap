const tf = require('@tensorflow/tfjs');

import { 
    euclidean,

} from './utils';

export default class UMAP {
  
    /**
     * @param {int} nNeighbors The size of the local neighborhood used for manifold approximation. 
     *                         Larger values result in more global views of the manifold, while smaller
     *                         values result in more local data being preserved. In general, values should
     *                         be in the range 2 to 100. Default: 15
     * @param {int} nComponents The dimension of the space to embed into. This defaults to 2 to provide easy
     *                          visualization, but can reasonably be set to any integer value in the range 2
     *                          to 100. Default: 2
     * @param {function} metric The metric to use to compute distances in high dimensional space. 
     *                          Should be a function that takes two 1d arrays and returns a float. 
     *                          Default: euclidean
     * @param {array} metricArgs Optional arguments to pass on to the metric, such as the p value for Minkowski
     *                           distance.
     * @param {int} nEpochs The number of training epochs to be used in optimizing the low-dimensional embedding.
     *                      Larger values result in more accurate embeddings. If undefined a value will be selected
     *                      based on the size of the input dataset. (200 for large datasets, 500 for small).
     * @param {float} learningRate The initial learning rate for the embedding optimization. Default: 1.0
     * @param {string} init How to initialize the low dimensional embedding. Options are:
     *                      - 'spectral': use a spectral embedding of the fuzzy 1-skeleton
     *                      - 'random': assign initial embedding positions at random.
     * @param {float} minDist The effective minimum distance between embedded points. Smaller values will result
     *                        in a more clustered/clumped embedding where nearby points on the manifold are drawn
     *                        closer together, while larger values will result in a more even dispersal of points.
     *                        The value should be set relative to the spread value, which determines the scale at
     *                        which embedded points will be spread out. Default: 0.1
     * @param {float} spread The effective scale of embedded points. In combination with minDist, this determines
     *                       how clustered/clumped the embedded points are. Default: 1.0
     * @param {float} setOpMixRatio Interpolate between (fuzzy) union and intersection as the set operation used
     *                              to combine local fuzzy simplicial sets to obtain a global fuzzy simplicial set.
     *                              Both fuzzy set operations use the product t-norm. The value of this parameter
     *                              should be between 0.0 and 1.0; a value of 1.0 will use a pure fuzzy union, 
     *                              while 0.0 will use a pure fuzzy intersection. Default: 1.0
     * @param {int} localConnectivity The local connectivity required -- i.e. the number of nearest neighbors that
     *                                should be assumed to be connected at a local level. The higher this value,
     *                                the more connected the manifold becomes locally. In practice this should not
     *                                be more than the local intrinsic dimension of the manifold. Default: 1
     * @param {float} repulsionStrength Weighting applied to negative samples in low dimensional embedding
     *                                  optimization. Values higher than one will result in greater weight being
     *                                  given to negative samples. Default: 1.0
     * @param {int} negativeSampleRate The number of negative samples to select per positive sample in the
     *                                 optimization process. Increasing this value will result in greater repulsive
     *                                 force being applied, greater optimization cost, but slightly more accuracy.
     *                                 Default: 5
     * @param {float} transformQueueSize For transform operations (embedding new points using a trained model) this
     *                                   this will control how aggressively to search for nearest neighbors. Larger
     *                                   values will result in slower performance but more accurate nearest neighbor
     *                                   evaluation. Default: 4.0
     * @param {float} a More specific parameters controlling the embedding. If undefined these values are set
     *                  automatically as determined by minDist and spread.
     * @param {float} b More specific parameters controlling the embedding. If undefined these values are set
     *                  automatically as determined by minDist and spread.
     * @param {boolean} angularRPForest Whether to use an angular random projection forest to initialize the
     *                                  approximate nearest neighbor search. This can be faster, but is mostly useful
     *                                  for metrics that use an angular style distance such as cosine, correlation, 
     *                                  etc. Default: false
     */
    constructor(config) {
      let { 
        nNeighbors,
        nComponents,
        metric,
        metricArgs,
        nEpochs,
        learningRate,
        init,
        minDist,
        spread,
        setOpMixRatio,
        localConnectivity,
        repulsionStrength,
        negativeSampleRate,
        transformQueueSize,
        a,
        b,
        angularRPForest
      } = config || {};
      this.nNeighbors = nNeighbors || 15;
      this.nComponents = nComponents || 2; // must be greater than 0
      this.nEpochs = nEpochs; // if defined, must be a positive integer greater than 10
      this.metric = metric || euclidean;
      this.metricArgs = metricArgs || [];
      this.learningRate = learningRate || 1.0;
      this.init = init || "spectral";
      this.minDist = minDist || 0.1; // must be less than or equal to spread, must be greater than 0.0
      this.spread = spread || 1.0;
      this.setOpMixRatio = setOpMixRatio || 1.0; // must be between 0.0 and 1.0
      this.localConnectivity = localConnectivity || 1.0;
      this.repulsionStrength = repulsionStrength || 1.0; // cannot be negative
      this.negativeSampleRate = negativeSampleRate || 5;
      this.transformQueueSize = transformQueueSize || 4.0;
      this.a = a;
      this.b = b;
      this.angularRPForest = angularRPForest || false;
    }
    
    /**
     * Fit X into an embedded space and return that transformed output.
     * @param {array} X (nSamples, nFeatures) The input data.
     * @returns {object} The computed embedding.
     */
    async fitTransform(X) {
      await this.fit(X);
      return this.embedding_
    }
    
    /**
     * Fit X into an embedded space.
     * @param {array} X (nSamples, nFeatures) The input data.
     */
    async fit(X) {
      this._rawData = X;
      if(this.a === undefined || this.b === undefined) {
        let { a, b } = await findABParams(this.spread, this.minDist);
        this.a = a;
        this.b = b;
      }
      this._initialAlpha = this.learningRate;
      
      // Error check nNeighbor based on data size
      const tX = tf.tensor(X);
      if(tX.shape[0] <= this.nNeighbors) {
        if(tX.shape[0] == 1) {
          self.embedding_ = tf.zeros([1, this.nComponents]);
          return this;
        }
        // n_neighbors is larger than the dataset size; truncating to X.shape[0] - 1
        this._nNeighbors = tX.shape[0] - 1;
      } else {
        this._nNeighbors = this.nNeighbors;
      }
      
      // Construct fuzzy simplicial set
      let { KNNIndices, KNNDists, RPForest } = await nearestNeighbors(
        tX, 
        this._nNeighbors,
        this.metric,
        this.metricArgs,
        this.angularRPForest
      );
      this._KNNIndices = KNNIndices;
      this._KNNDists = KNNDists;
      this._RPForest = RPForest;
      /*this.graph_ = fuzzySimplicialSet(
        tX,
        this._nNeighbors,
        this.metric,
        this._KNNIndices,
        this._KNNDists,
        this.angularRPForest,
        this.setOpMixRatio,
        this.localConnectivity
      );*/
    }
    
    
  }