import * as tf from "@tensorflow/tfjs";
import * as blazeface from "@tensorflow-models/blazeface";
import "./style.css";
import { STOP_DATA_GATHER } from "./lib/constants";
import loadMobileNetModel from "./lib/loadMobileNetModel";
import prepareModel from "./lib/prepareModel";
import getWebcamStream from "./lib/getWebcamStream";
import calculateFeaturesOnCurrentFrame from "./lib/calculateFeaturesOnCurrentFrame";

const ENABLE_CAM_BUTTON = document.getElementById(
	"enableCam"
) as HTMLButtonElement;
const RESET_BUTTON = document.getElementById("reset") as HTMLButtonElement;
const TRAIN_BUTTON = document.getElementById("train") as HTMLButtonElement;
const TRAINING_PROGRESS_BAR = document.getElementById(
	"training-progress"
) as HTMLProgressElement;
const TRAINING_PROGRESS_BAR_LABEL = document.getElementById(
	"training-label"
) as HTMLLabelElement;
const TRAINING_OUTPUT = document.getElementById("training") as HTMLDivElement;
const DATA_BUTTONS_CONTAINER = document.getElementById(
	"data-buttons"
) as HTMLDivElement;
const VIDEO = document.getElementById("webcam") as HTMLVideoElement;
const CANVAS = document.getElementById("canvas") as HTMLCanvasElement;

const CLASS_NAMES: string[] = [];

ENABLE_CAM_BUTTON.addEventListener("click", enableCam);
TRAIN_BUTTON.addEventListener("click", trainAndPredict);
RESET_BUTTON.addEventListener("click", reset);

function attachListeners() {
	DATA_BUTTONS_CONTAINER.querySelectorAll("button").forEach(button => {
		button.addEventListener("mousedown", gatherDataForClass);
		button.addEventListener("mouseup", gatherDataForClass);
		button.addEventListener("touchend", gatherDataForClass);

		CLASS_NAMES.push(button.getAttribute("data-name") as string);
	});
}

attachListeners();

let gatherDataState = STOP_DATA_GATHER;
let videoPlaying = false;
let predict = false;

const trainingDataInputs: any[] = [];
const trainingDataOutputs: any[] = [];
const examplesCount: number[] = [];

let mobilenet: Awaited<ReturnType<typeof loadMobileNetModel>>;

async function loadModel() {
	mobilenet = await loadMobileNetModel();
}

loadModel();

logStatus("MobileNet v3 loaded successfully!");

const model = tf.sequential();

prepareModel(model, CLASS_NAMES.length);

function gatherDataForClass(this: HTMLButtonElement) {
	const classNumber = parseInt(this.getAttribute("data-1hot") as string);
	gatherDataState =
		gatherDataState === STOP_DATA_GATHER ? classNumber : STOP_DATA_GATHER;
	loopDataGathering();
}

function loopDataGathering() {
	if (!videoPlaying || gatherDataState === STOP_DATA_GATHER) return;

	const imageFeatures = calculateFeaturesOnCurrentFrame(mobilenet);

	trainingDataInputs.push(imageFeatures);
	trainingDataOutputs.push(gatherDataState);

	const current = examplesCount[gatherDataState];

	examplesCount[gatherDataState] = current === undefined ? 1 : current + 1;

	const text = CLASS_NAMES.reduce((acc, className, index) => {
		return acc + `${className} data count: ${examplesCount[index]}. `;
	}, "");

	logStatus(text);

	window.requestAnimationFrame(loopDataGathering);
}

async function trainAndPredict() {
	predict = false;
	TRAINING_OUTPUT.classList.remove("removed");
	tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

	const outputsAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
	const oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
	const inputsAsTensor = tf.stack(trainingDataInputs);

	await model.fit(inputsAsTensor, oneHotOutputs, {
		shuffle: true,
		batchSize: 5,
		epochs: 10,
		callbacks: { onEpochEnd: logProgress },
	});

	outputsAsTensor.dispose();
	oneHotOutputs.dispose();
	inputsAsTensor.dispose();

	predict = true;
	predictLoop();
}

function logProgress(epoch: number) {
	TRAINING_PROGRESS_BAR.value = epoch;
	TRAINING_PROGRESS_BAR_LABEL.innerText = `${epoch + 1}/10`;
}

function predictLoop() {
	if (!predict) return;

	tf.tidy(() => {
		const imageFeatures = calculateFeaturesOnCurrentFrame(mobilenet);
		// @ts-ignore
		const prediction = model.predict(imageFeatures.expandDims()).squeeze();
		const highestIndex = prediction.argMax().arraySync();
		const predictionArray = prediction.arraySync();

		logStatus(
			`Prediction: ${CLASS_NAMES[highestIndex]} with ${Math.round(
				predictionArray[highestIndex] * 100
			)}% confidence`
		);
	});

	window.requestAnimationFrame(predictLoop);
}

function reset() {
	predict = false;
	examplesCount.splice(0);
	trainingDataInputs.forEach(dataInput => dataInput.dispose());
	trainingDataInputs.splice(0);
	trainingDataOutputs.splice(0);

	logStatus("No data collected");
}

function logStatus(text: string) {
	document.getElementById("status")!.innerText = text;
}

let faceModel: any;
const ctx = CANVAS.getContext("2d") as any;

async function detectFaces() {
	const prediction = await faceModel.estimateFaces(VIDEO, false);

	ctx.drawImage(VIDEO, 0, 0, 640, 480);

	prediction.forEach(pred => {
		ctx.beginPath();
		ctx.lineWidth = 4;
		ctx.strokeStyle = "#2e79f2";
		ctx.rect(
			pred.topLeft[0],
			pred.topLeft[1],
			pred.bottomRight[0] - pred.topLeft[0],
			pred.bottomRight[1] - pred.topLeft[1]
		);
		ctx.stroke();
	});
}

function enableCam() {
	getWebcamStream()
		.then(async () => {
			faceModel = await blazeface.load();
			setInterval(detectFaces, 50);
		})
		.then(() => {
			videoPlaying = true;
			ENABLE_CAM_BUTTON.classList.add("removed");
		});
}
