(function (factory) {
	typeof define === 'function' && define.amd ? define(factory) :
	factory();
}(function () { 'use strict';

	//const tf = require('@tensorflow/tfjs');
	//const mnist = require("mnist")

	console.log(tf.tensor([1, 2, 3]).data());
	//console.log(mnist);

}));
