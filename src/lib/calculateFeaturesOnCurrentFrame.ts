import * as tf from "@tensorflow/tfjs";
import { MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH } from "./constants";

const VIDEO = document.getElementById("webcam") as HTMLVideoElement;

export default function calculateFeaturesOnCurrentFrame(
	mobilenet: tf.GraphModel
) {
	return tf.tidy(() => {
		const videoFrameAsTensor = tf.browser.fromPixels(VIDEO);
		const resizedTensorFrame = tf.image.resizeBilinear(
			videoFrameAsTensor,
			[MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
			true
		);

		const normalizedTensorFrame = resizedTensorFrame.div(255);

		// @ts-ignore
		return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
	});
}
