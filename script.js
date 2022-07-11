// let images = [img1, img2, img3, img4]
const TRAIN_BUTTON = document.getElementById('train');
const PREDICT_BUTTON = document.getElementById('predict');
const STATUS = document.getElementById('status');

TRAIN_BUTTON.addEventListener('click', trainAndPredict);
PREDICT_BUTTON.addEventListener('click', predictNew);

const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;
const STOP_DATA_GATHER = -1;

let mobilenet = undefined;
let model = undefined;
let images = [];
//let gatherDataState = STOP_DATA_GATHER;
let trainingDataInputs = [];
let trainingDataOutputs = [];
let examplesCount = [];
let predict = false;

let CLASS_NAMES = []



async function loadMobileNetFeatureModel() {
    const URL =
        'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1';

    mobilenet = await tf.loadGraphModel(URL, { fromTFHub: true });
    STATUS.innerText = 'MobileNet v3 loaded successfully!';

    // Warm up the model by passing zeros through it once.
    tf.tidy(function () {
        let answer = mobilenet.predict(tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3]));
        console.log(answer.shape);
    });
    let elements = document.getElementsByTagName("*");
    for (let s = 0; s < elements.length; s++) {
        if (elements[s].id == 'train') break;
        elements[s].disabled = false;
        console.log();
    }

}

// Call the function immediately to start loading.
loadMobileNetFeatureModel();
// setTimeout(() => {
//     dataGatherLoop();
// }, 20000);




function dataGatherLoop() {
    let len = document.getElementById("classes").childElementCount;
    let n = 0, m = 0;
    for (n = 0; n < len; n++) {
        CLASS_NAMES[n] = document.getElementById("classes").children[n].children[1].innerText;
    }
    console.log(CLASS_NAMES);
    model = tf.sequential();
    model.add(tf.layers.dense({ inputShape: [1024], units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: CLASS_NAMES.length, activation: 'softmax' }));
    model.summary();

    // Compile the model with the defined optimizer and specify a loss function to use.
    model.compile({
        // Adam changes the learning rate over time which is useful.
        optimizer: 'adam',
        // Use the correct loss function. If 2 classes of data, must use binaryCrossentropy.
        // Else categoricalCrossentropy is used if more than 2 classes.
        loss: (CLASS_NAMES.length === 2) ? 'binaryCrossentropy' : 'categoricalCrossentropy',
        // As this is a classification problem you can record accuracy in the logs too!
        metrics: ['accuracy']
    });
    for (let n = 0; n < len; n++) {
        let dataCount = document.getElementsByClassName("inputs" + n).length;
        console.log(dataCount);
        for (let m = 0; m < dataCount; m++) {
            console.log(n, m);
            let imageFeatures = tf.tidy(function () {
                let videoFrameAsTensor = tf.browser.fromPixels(document.getElementsByClassName("inputs" + n)[m]);
                let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT,
                    MOBILE_NET_INPUT_WIDTH], true);
                let normalizedTensorFrame = resizedTensorFrame.div(255);
                return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
            });

            trainingDataInputs.push(imageFeatures);
            trainingDataOutputs.push(n);

            // Intialize array index element if currently undefined.
            if (examplesCount[n] === undefined) {
                examplesCount[n] = 0;
            }
            examplesCount[n]++;

            STATUS.innerText = '';
            for (let n = 0; n < CLASS_NAMES.length; n++) {
                STATUS.innerText += CLASS_NAMES[n] + ' data count: ' + examplesCount[n] + '. ';
            }
        }
    }
}



async function trainAndPredict() {
    dataGatherLoop();
    predict = false;
    tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
    let outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
    let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
    let inputsAsTensor = tf.stack(trainingDataInputs);

    let results = await model.fit(inputsAsTensor, oneHotOutputs, {
        shuffle: true, batchSize: 5, epochs: 10,
        callbacks: { onEpochEnd: logProgress }
    });

    outputsAsTensor.dispose();
    oneHotOutputs.dispose();
    inputsAsTensor.dispose();
    predict = true;
    console.log("done");
    document.getElementById('file').disabled = false;
    document.getElementById('predict').disabled = false;
}

async function predictNew() {
    if (predict) {
        tf.tidy(function () {
            let image = document.getElementById('inputs');
            let videoFrameAsTensor = tf.browser.fromPixels(image).div(255);
            let resizedTensorFrame = tf.image.resizeBilinear(videoFrameAsTensor, [MOBILE_NET_INPUT_HEIGHT,
                MOBILE_NET_INPUT_WIDTH], true);

            let imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims());
            let prediction = model.predict(imageFeatures).squeeze();
            let highestIndex = prediction.argMax().arraySync();
            let predictionArray = prediction.arraySync();
            console.log(highestIndex);
            console.log(predictionArray);
            STATUS.innerText = 'Prediction: ' + CLASS_NAMES[highestIndex] + ' with ' + Math.floor(predictionArray[highestIndex] * 100) + '% confidence';
        });
    }
}

function logProgress(epoch, logs) {
    console.log('Data for epoch ' + epoch, logs);
}

