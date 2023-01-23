import hasGetUserMedia from "./hasGetUserMedia";

const VIDEO = document.getElementById("webcam") as HTMLVideoElement;

export default async function getWebcamStream() {
	if (!hasGetUserMedia()) {
		console.warn("getUserMedia() is not supported by your browser");
	}

	const constraints = {
		video: true,
		width: 640,
		height: 480,
	};

	const stream = await navigator.mediaDevices.getUserMedia(constraints);

	VIDEO.srcObject = stream;

	return new Promise(resolve => VIDEO.addEventListener("loadeddata", resolve));
}
