import React, { useEffect, useRef, useState } from 'react';
import { mockUsers } from './mockUsers';
import * as faceapi from 'face-api.js';

export const App = () => {
  const [initializing, setInitializing] = useState(false);
  const [showImage, setShowImage] = useState('');
  const [countFaces, setCountFaces] = useState(0);
  const imageUpload = useRef();

  useEffect(() => {
    const loadModels = async () => {
      const MODEL_URL = process.env.PUBLIC_URL + '/models';
      setInitializing(true);
      Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      ]).then(handleStart);
    };

    loadModels();
  }, []);

  const handleStart = async () => {
    const Container = document.getElementsByClassName('Container');
    const CanvasContainer = document.createElement('div');
    CanvasContainer.style.position = 'relative';

    Container[0].append(CanvasContainer);

    const labeledDescriptors = await loadLabeledImages();

    const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.49);

    if (imageUpload.current.files.length > 0) {
      const image = await faceapi.bufferToImage(imageUpload.current.files[0]);

      setShowImage(image.src);
      setInitializing(false);

      const canvas = faceapi.createCanvasFromMedia(image);
      CanvasContainer.append(canvas);

      const displaySize = { width: image.width, height: image.height };

      faceapi.matchDimensions(canvas, displaySize);

      const detections = await faceapi
        .detectAllFaces(image)
        .withFaceLandmarks()
        .withFaceDescriptors();

      setCountFaces(detections.length);

      const resizedDetections = faceapi.resizeResults(detections, displaySize);

      const result = resizedDetections.map((detected) =>
        faceMatcher.findBestMatch(detected.descriptor)
      );

      result.forEach((result, i) => {
        const box = resizedDetections[i].detection.box;
        const drawBox = new faceapi.draw.DrawBox(box, {
          label: result.toString(),
        });
        drawBox.draw(canvas);
      });
    }
  };

  const loadLabeledImages = () => {
    return Promise.all(
      mockUsers.map(async (user) => {
        const label = user.name;
        const descriptions = [];

        for (let i = 0; i <= 1; i++) {
          const img = await faceapi.fetchImage(`${user.img[i]}`);

          const detections = await faceapi
            .detectSingleFace(img)
            .withFaceLandmarks()
            .withFaceDescriptor();

          descriptions.push(detections.descriptor);
        }

        return new faceapi.LabeledFaceDescriptors(label, descriptions);
      })
    );
  };

  return (
    <div className="MainContainer">
      <div className="Container">
        <h1>Face recognition</h1>
        <span>{initializing ? 'Waiting image...' : 'Image upload'}</span>
        <br />
        <input type="file" ref={imageUpload} onChange={handleStart} />
        <br />
        {showImage ? (
          <figure className="ImgContainer">
            <img src={showImage} />
          </figure>
        ) : null}

        {countFaces ? <h2>Total faces: {countFaces}</h2> : null}
      </div>
    </div>
  );
};
