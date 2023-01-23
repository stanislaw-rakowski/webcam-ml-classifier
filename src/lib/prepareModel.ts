import * as tf from "@tensorflow/tfjs";

export default function prepareModel(model: tf.Sequential, classes: number) {
	model.add(
		tf.layers.dense({ inputShape: [1024], units: 128, activation: "relu" })
	);
	model.add(tf.layers.dense({ units: classes, activation: "softmax" }));

	model.summary();

	model.compile({
		optimizer: "adam",
		loss: classes === 2 ? "binaryCrossentropy" : "categoricalCrossentropy",
		metrics: ["accuracy"],
	});
}
