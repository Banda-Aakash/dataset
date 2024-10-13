import React, { useEffect, useRef } from "react";

const CameraFeed = () => {
  const videoRef = useRef(null);

  useEffect(() => {
    // Access the user's webcam
    const getVideo = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (error) {
        console.error("Error accessing camera: ", error);
      }
    };

    getVideo();

    // Cleanup function to stop the video stream when the component unmounts
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject;
        const tracks = stream.getTracks();

        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  return (
    <div>
      <h2>Camera Feed</h2>
      <video ref={videoRef} width="600" height="400" autoPlay />
    </div>
  );
};

export default CameraFeed;
