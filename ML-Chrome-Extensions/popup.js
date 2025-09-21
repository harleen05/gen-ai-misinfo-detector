// popup.js

document.addEventListener("DOMContentLoaded", () => {
    const detectionTypeSelect = document.getElementById("detection-type");
    const fileUpload = document.getElementById("file-upload");
    const fileNameSpan = document.getElementById("file-name");
    const fileInstruction = document.getElementById("file-instruction");
    const runDetectionBtn = document.getElementById("run-detection-btn");

    let selectedFile = null;

    // Update file name display when file is selected
    fileUpload.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            fileNameSpan.textContent = selectedFile.name;
            runDetectionBtn.disabled = false;
        } else {
            selectedFile = null;
            fileNameSpan.textContent = "No file chosen";
            runDetectionBtn.disabled = true;
        }
    });

    // Update file instructions based on detection type
    detectionTypeSelect.addEventListener("change", (e) => {
        const type = e.target.value;
        switch (type) {
            case "audio":
                fileInstruction.textContent = "Select an audio file (mp3, wav):";
                fileUpload.accept = ".mp3,.wav";
                break;
            case "video":
                fileInstruction.textContent = "Select a video file (mp4, mov):";
                fileUpload.accept = ".mp4,.mov";
                break;
            case "text":
                fileInstruction.textContent = "Select a text file (txt):";
                fileUpload.accept = ".txt";
                break;
        }
        selectedFile = null;
        fileNameSpan.textContent = "No file chosen";
        runDetectionBtn.disabled = true;
    });

    // Handle Run Detection button click
    runDetectionBtn.addEventListener("click", async () => {
        if (!selectedFile) {
            alert("Please select a file first.");
            return;
        }

        runDetectionBtn.disabled = true;
        runDetectionBtn.textContent = "Detecting...";

        const detectionType = detectionTypeSelect.value;
        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("type", detectionType);

        try {
            const response = await fetch("https://your-ml-api.com/detect", {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error("Network response was not ok");

            const result = await response.json();
            alert(`Detection Result: ${JSON.stringify(result)}`);
        } catch (error) {
            console.error("Error:", error);
            alert("Error detecting file. Please try again.");
        } finally {
            runDetectionBtn.disabled = false;
            runDetectionBtn.textContent = "Run Detection";
        }
    });
});


