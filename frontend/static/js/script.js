//SPA
function showContent(sectionId) {
  const sections = document.querySelectorAll(".content-section");

  // Hide all content sections
  sections.forEach((section) => {
    section.style.display = "none";
  });

  // Show the selected section by its ID
  const selectedSection = document.getElementById(sectionId);
  if (selectedSection) {
    selectedSection.style.display = "block";
  }
}

// (HOME SECTION) drag and drop image
const dropArea = document.getElementById("drop-area");
const inputFile = document.getElementById("input-file");
const customButton = dropArea.querySelector("button");
//const fileNameDisplay = dropArea.querySelector("span");
//const identifyButton = document.getElementById("identify-button"); // The "Identify" button

customButton.addEventListener("click", () => inputFile.click()); // Trigger file input when the button is clicked
inputFile.addEventListener("change", handleFileUpload); // Handle file selection
dropArea.addEventListener("dragover", (e) => e.preventDefault()); // Allow dragover
dropArea.addEventListener("drop", handleDrop); // Handle file drop

async function handleFileUpload() {
  if (inputFile.files.length > 0) {
    const file = inputFile.files[0];
    // Create a FormData object to send the image
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      console.log(data);

      // Check if the prediction and other fields are present in the response
      if (data.baseline_prediction && data.proposed_prediction) {
        displayResult(data);
        showContent("result");
      } else {
        alert("No prediction received.");
      }
    } catch (error) {
      console.error("Error:", error);
      alert("Error uploading image.");
    }
  } else {
    resetFileDisplay();
  }
}

function displayResult(data) {
  const resultContainer = document.getElementById("result-container");
  resultContainer.innerHTML = `
    <h3>Baseline Prediction: ${data.baseline_prediction}</h3>
    <p>Confidence: ${data.baseline_confidence}</p>
    <p>Prediction Time: ${data.baseline_prediction_time} seconds</p>
    <h3>Proposed Prediction: ${data.proposed_prediction}</h3>
    <p>Confidence: ${data.proposed_confidence}</p>
    <p>Prediction Time: ${data.proposed_prediction_time} seconds</p>
    <h4>Masked Image:</h4>
    <img src="data:image/png;base64,${data.image}" alt="Predicted Image" />
    <img src="data:image/png;base64,${data.masked_image}" alt="Predicted Image" />
  `;
}

function handleDrop(e) {
  e.preventDefault();
  inputFile.files = e.dataTransfer.files; // Assign dropped files to input
  handleFileUpload(); // Trigger file handling
}

function resetFileDisplay() {
  fileNameDisplay.textContent = "No file chosen";

  // Hide the "Identify" button if no file is uploaded
  //identifyButton.style.display = "none";
}

// (RESULT SECTION)
function toggleAdditionalContent() {
  var content = document.getElementById("additional-content");
  // Toggle the display style between 'none' and 'block'
  if (content.style.display === "none") {
    content.style.display = "block";
  } else {
    content.style.display = "none";
  }
}

// GRAPH
// Ensure this runs after the DOM is fully loaded
document.addEventListener("DOMContentLoaded", function () {
  // Get the canvas element
  const ctx = document.getElementById("topSpeciesChart").getContext("2d");

  // Example data (replace this with your dynamic data)
  const speciesLabels = [
    "Species A",
    "Species B",
    "Species C",
    "Species D",
    "Species E",
  ];
  const matchScores = [100, 75, 65, 55, 50]; // Match percentages

  // Create the bar chart
  const topSpeciesChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: speciesLabels, // X-axis labels
      datasets: [
        {
          label: "Match Percentage (%)",
          data: matchScores, // Y-axis values
          backgroundColor: [
            "rgba(75, 192, 192, 0.6)",
            "rgba(54, 162, 235, 0.6)",
            "rgba(255, 206, 86, 0.6)",
            "rgba(255, 99, 132, 0.6)",
            "rgba(153, 102, 255, 0.6)",
          ],
          borderColor: [
            "rgba(75, 192, 192, 1)",
            "rgba(54, 162, 235, 1)",
            "rgba(255, 206, 86, 1)",
            "rgba(255, 99, 132, 1)",
            "rgba(153, 102, 255, 1)",
          ],
          borderWidth: 1,
          barThickness: 40,
          maxBarThickness: 60,
        },
      ],
    },
    options: {
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Match Percentage (%)",
          },
        },
        x: {
          title: {
            display: true,
            text: "Species",
          },
        },
      },
      responsive: true,
      plugins: {
        legend: {
          display: true,
          position: "top",
        },
        tooltip: {
          enabled: true,
        },
      },
    },
  });
});
