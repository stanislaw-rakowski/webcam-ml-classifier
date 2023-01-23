import * as tf from "@tensorflow/tfjs";
import { MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH } from "./constants";

const MODEL_URL =
	"https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1";

export default async function loadMobileNetModel() {
	const mobilenet = await tf.loadGraphModel(MODEL_URL, { fromTFHub: true });

	tf.tidy(() => {
		mobilenet.predict(
			tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3])
		);
	});

	return mobilenet;
}
