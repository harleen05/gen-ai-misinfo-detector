document.addEventListener("DOMContentLoaded", () => {
    const detectionTypeSelect = document.getElementById("detection-type");
    const fileUpload = document.getElementById("file-upload");
    const fileNameSpan = document.getElementById("file-name");
    const fileInstruction = document.getElementById("file-instruction");
    const runDetectionBtn = document.getElementById("run-detection-btn");
    const resultDiv = document.getElementById("result"); // add <div id="result"></div> in popup.html

    let selectedFile = null;

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
            // 1️⃣ Call ML detection backend
            const response = await fetch("https://your-ml-api.com/detect", {
                method: "POST",
                body: formData
            });

            if (!response.ok) throw new Error("Network response was not ok");
            const modelOutputs = await response.json();

            // 2️⃣ Send detection results to explanation layer
            const explanationRes = await fetch("http://127.0.0.1:5001/generate_explanation", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    model_outputs: modelOutputs,
                    mode: "student" // or "expert"
                })
            });

            if (!explanationRes.ok) throw new Error("Explanation service error");
            const explanation = await explanationRes.json();

            // 3️⃣ Show explanation nicely in popup
            resultDiv.innerHTML = `
                <h3>Detection Result</h3>
                <pre>${JSON.stringify(modelOutputs, null, 2)}</pre>
                <h3>Explanation</h3>
                <p>${explanation.explanation}</p>
                <small>Confidence: ${explanation.confidence}</small>
            `;

        } catch (error) {
            console.error("Error:", error);
            alert("Error detecting file or generating explanation.");
        } finally {
            runDetectionBtn.disabled = false;
            runDetectionBtn.textContent = "Run Detection";
        }
    });
});