import React, { useState } from 'react';

const API_BASE = "https://deepfake-api-217279920936.us-central1.run.app";

const DeepfakeDetector = () => {
  const [detectionType, setDetectionType] = useState('audio');
  const [selectedFile, setSelectedFile] = useState(null);
  const [textInput, setTextInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);

  const getFileInstruction = () => {
    switch (detectionType) {
      case 'audio':
        return 'Select an audio file (mp3, wav):';
      case 'video':
        return 'Select a video file (mp4, avi, mov):';
      case 'text':
        return 'Paste your text below or select a text file:';
      default:
        return 'Select a file:';
    }
  };

  const getAcceptedFileTypes = () => {
    switch (detectionType) {
      case 'audio':
        return '.mp3,.wav';
      case 'video':
        return '.mp4,.avi,.mov';
      case 'text':
        return '.txt,.doc,.docx';
      default:
        return '*';
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
  };

  const handleDetectionTypeChange = (event) => {
    setDetectionType(event.target.value);
    setSelectedFile(null); // Clear selected file when changing detection type
    setTextInput(''); // Clear text input when changing detection type
  };

  const handleRunDetection = async () => {
    if (detectionType === 'text' && !textInput.trim() && !selectedFile) {
      alert('Please enter text or select a text file');
      return;
    } else if (detectionType !== 'text' && !selectedFile) {
      alert('Please select a file first');
      return;
    }

    setIsProcessing(true);
    
    try {
      // API call to your Google Cloud deployed Flask backend
      const formData = new FormData();
      
      if (detectionType === 'text' && textInput.trim()) {
        formData.append('text', textInput);
      } else if (selectedFile) {
        formData.append('file', selectedFile);
      }
      
      formData.append('detection_type', detectionType);

      const response = await fetch(`${API_BASE}/api/detect`, {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        // Handle successful response
        console.log('Detection result:', result);
        alert(`Detection completed! Result: ${JSON.stringify(result)}`);
      } else {
        throw new Error('Detection failed');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('Error occurred during detection. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4 sm:p-6 lg:p-8">
      <div className="w-full max-w-xs sm:max-w-sm md:max-w-md lg:max-w-lg xl:max-w-xl bg-white rounded-xl shadow-lg border border-gray-200 p-4 sm:p-6 lg:p-8">
        <header className="text-center mb-4 sm:mb-6 border-b border-gray-200 pb-3 sm:pb-4">
          <h1 className="text-xl sm:text-2xl lg:text-3xl font-semibold text-gray-800 flex items-center justify-center flex-wrap">
            <span className="text-2xl sm:text-3xl lg:text-4xl mr-2">🛡️</span>
            <span className="break-words">Deepfake Detector</span>
          </h1>
        </header>

        <section className="mb-4 sm:mb-6">
          <p className="mb-2 font-medium text-gray-600 text-sm sm:text-base">Select a detection type:</p>
          <div className="relative">
            <select
              id="detection-type"
              value={detectionType}
              onChange={handleDetectionTypeChange}
              className="w-full p-2 sm:p-3 border border-gray-300 rounded-lg bg-gray-50 text-sm sm:text-base cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-200 focus:border-blue-500 transition-all"
            >
              <option value="audio">Audio Detection</option>
              <option value="video">Video Detection</option>
              <option value="text">Text Detection</option>
            </select>
          </div>
        </section>

        <section className="mb-4 sm:mb-6">
          <p className="mb-2 font-medium text-gray-600 text-sm sm:text-base">{getFileInstruction()}</p>
          
          {detectionType === 'text' ? (
            <div className="space-y-3 sm:space-y-4">
              {/* Text Input Area */}
              <textarea
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                placeholder="Enter your text here for deepfake detection..."
                className="w-full p-2 sm:p-3 border border-gray-300 rounded-lg bg-gray-50 text-sm sm:text-base resize-none focus:outline-none focus:ring-2 focus:ring-blue-200 focus:border-blue-500 transition-all"
                rows="3"
              />
              
              {/* OR divider */}
              <div className="flex items-center">
                <div className="flex-grow border-t border-gray-300"></div>
                <span className="px-2 sm:px-3 text-xs sm:text-sm text-gray-500 bg-white">OR</span>
                <div className="flex-grow border-t border-gray-300"></div>
              </div>
              
              {/* File upload for text */}
              <div className="flex flex-col sm:flex-row items-start sm:items-center gap-2 sm:gap-3">
                <label
                  htmlFor="file-upload"
                  className="bg-blue-500 text-white px-3 sm:px-4 py-2 sm:py-3 rounded-lg cursor-pointer font-semibold hover:bg-blue-600 active:transform active:translate-y-0.5 transition-all whitespace-nowrap text-sm sm:text-base w-full sm:w-auto text-center"
                >
                  Choose Text File
                </label>
                <input
                  type="file"
                  id="file-upload"
                  accept={getAcceptedFileTypes()}
                  onChange={handleFileChange}
                  className="hidden"
                />
                <span className="text-xs sm:text-sm text-gray-600 flex-grow overflow-hidden text-ellipsis whitespace-nowrap w-full sm:w-auto">
                  {selectedFile ? selectedFile.name : 'No file chosen'}
                </span>
              </div>
            </div>
          ) : (
            /* File upload for audio/video */
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-2 sm:gap-3">
              <label
                htmlFor="file-upload"
                className="bg-blue-500 text-white px-3 sm:px-4 py-2 sm:py-3 rounded-lg cursor-pointer font-semibold hover:bg-blue-600 active:transform active:translate-y-0.5 transition-all whitespace-nowrap text-sm sm:text-base w-full sm:w-auto text-center"
              >
                Choose File
              </label>
              <input
                type="file"
                id="file-upload"
                accept={getAcceptedFileTypes()}
                onChange={handleFileChange}
                className="hidden"
              />
              <span className="text-xs sm:text-sm text-gray-600 flex-grow overflow-hidden text-ellipsis whitespace-nowrap w-full sm:w-auto">
                {selectedFile ? selectedFile.name : 'No file chosen'}
              </span>
            </div>
          )}
        </section>

        <button
          onClick={handleRunDetection}
          disabled={isProcessing || (detectionType === 'text' ? (!textInput.trim() && !selectedFile) : !selectedFile)}
          className={`w-full p-2 sm:p-3 rounded-lg text-sm sm:text-base font-semibold transition-all mt-3 ${
            isProcessing || (detectionType === 'text' ? (!textInput.trim() && !selectedFile) : !selectedFile)
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-green-500 hover:bg-green-600 active:transform active:translate-y-0.5 text-white cursor-pointer'
          }`}
        >
          {isProcessing ? 'Processing...' : 'Run Detection'}
        </button>

        <div className="mt-4 sm:mt-6 flex items-start sm:items-center text-xs sm:text-sm text-gray-500 bg-gray-50 p-2 sm:p-3 rounded-lg">
          <span className="text-base sm:text-lg mr-2 flex-shrink-0">💡</span>
          <span className="leading-relaxed">Tip: Keep files under 50 MB for faster results.</span>
        </div>
      </div>
    </div>
  );
};

export default DeepfakeDetector;