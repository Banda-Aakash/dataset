import React, { useState } from "react";
import axios from "axios";

const SignLanguageDetector = () => {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) {
      alert("Please upload an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://localhost:5000/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      setPrediction(response.data.prediction);
    } catch (error) {
      console.error("Error uploading image:", error);
      setPrediction("Error in prediction");
    }
  };

  return (
    <div style={{ textAlign: "center" }}>
      <h1>Sign Language Detector</h1>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      <button onClick={handleSubmit}>Predict</button>
      {prediction && <h2>Predicted Sign: {prediction}</h2>}
    </div>
  );
};

export default SignLanguageDetector;
