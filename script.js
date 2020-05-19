const imageUpload = document.getElementById('imageUpload')

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(start)


// Detecting the image and labeling
async function start() {
  // created container for box
  const container = document.createElement('div');
  container.style.position = 'relative';
  document.body.append(container);

  // calling the 
  const labeledFaceDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.faceMatcher(labeledFaceDescriptors,
    0.6);

  imageUpload.addEventListener('change', async () => {
    const image = await faceapi.bufferToImage(imageUpload.files[0]);

    // For drawing boxes on the image
    container.append(image);
    const canvas = faceapi.createCanvasFromMedia(image);
    container.append(canvas);

    // changing dimesions of the canvas
    const dimensions = { width: image.width, height: image.height };
    faceapi.matchDimensions(canvas, dimensions);


    const detections = await faceapi.detectAllFaces(image)
      .withFaceLandmarks().withFaceDescriptors()

    // resize all our detections for our dimensions
    const resizedDetections = await faceapi.resizeResults(detections, dimensions);

    const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));

    // drawing the actual box for each face
    results.forEach((results, i) => {
      const box = resizedDetections[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: results.toString()
      })
      drawBox.draw(canvas);
    });
  })
}

// function to parse all the names from the images
function loadLabeledImages() {
  const labels = ['jhalani']

  // return all the promises for returning all the images
  return Promise.all(
    labels.map(async label => {
      const descriptions = [];
      for (let i = 1; i <= 3; i++) {
        const img = await faceapi.fetchImage(
          `https://raw.githubusercontent.com/nightwarriorftw/face-detection-js/master/labeled_images/${label}/${i}.jpeg`
          );
        const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
        descriptions.push(detections.descriptor)
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  )
}