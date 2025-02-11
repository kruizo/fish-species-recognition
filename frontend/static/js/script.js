document.addEventListener("DOMContentLoaded", function () {
  const dropArea = document.getElementById("drop-area");
  const fileInput = document.getElementById("input-file");
  const uploadButton = document.getElementById("upload-btn");
  const loader = document.getElementById("loader");
  const contentSection = document.getElementById("content-section");
  const uploadSection = document.getElementById("upload-section");
  const resultSection = document.getElementById("result-section");
  uploadButton.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", handleFileUpload);
  dropArea.addEventListener("dragover", (e) => e.preventDefault());
  dropArea.addEventListener("drop", handleDrop);

  function handleDrop(e) {
    e.preventDefault();
    fileInput.files = e.dataTransfer.files;
    handleFileUpload();
  }

  async function handleFileUpload() {
    if (fileInput.files.length > 0) {
      toggleVisibility(loader, true, "flex");

      const file = fileInput.files[0];
      const formData = new FormData();
      formData.append("file", file);

      try {
        toggleVisibility(contentSection, false);

        const response = await fetch("/predict", {
          method: "POST",
          body: formData,
        });

        const data = await response.json();
        console.log(data);

        if (data.baseline_prediction && data.proposed_prediction) {
          displayResult(data);
          toggleVisibility(contentSection, true);
          toggleVisibility(uploadSection, false);
          toggleVisibility(resultSection, true);
          toggleVisibility(loader, false);
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

  function toggleVisibility(element, show, display = "block") {
    element.style.display = show ? display : "none";
  }

  function displayResult(data) {
    document.querySelector("#result-section").classList.remove("hidden");

    // Proposed
    document
      .querySelectorAll(".proposed-pred")
      .forEach((el) => (el.textContent = data.proposed_prediction));
    // Proposed
    document.querySelectorAll(".proposed-conf").forEach((el) => {
      el.textContent = `${(data.proposed_confidence * 100).toFixed(2)}`;
    });
    document.querySelectorAll(".proposed-speed").forEach((el) => {
      el.textContent = `${data.proposed_prediction_time.toFixed(2)}`;
    });

    // Baseline
    document.querySelectorAll(".baseline-pred").forEach((el) => {
      el.textContent = data.baseline_prediction;
    });
    document.querySelectorAll(".baseline-speed").forEach((el) => {
      el.textContent = `${data.baseline_prediction_time.toFixed(2)}`;
    });
    document.querySelectorAll(".baseline-conf").forEach((el) => {
      el.textContent = `${(data.baseline_confidence * 100).toFixed(2)}`;
    });

    // Set images
    const baselineImg = document.querySelector("#baseline-card img");
    const proposedImg = document.querySelector("#proposed-card img");

    baselineImg.src = `data:image/png;base64,${data.original_image}`;
    proposedImg.src = `data:image/png;base64,${data.masked_image}`;

    // Set card by highest
    const baselineConfidence = data.baseline_confidence * 100;
    const proposedConfidence = data.proposed_confidence * 100;
    const baselineCard = document.getElementById("baseline-card");
    const proposedCard = document.getElementById("proposed-card");

    if (baselineConfidence >= proposedConfidence) {
      baselineCard.className = "card";
      proposedCard.className = "card-plain";
      proposedCard.querySelectorAll("p span").forEach((el) => {
        el.classList.add("text-red-400");
      });
      baselineCard.querySelectorAll("p span").forEach((el) => {
        el.classList.add("text-green-400");
      });
    } else {
      baselineCard.className = "card-plain";
      proposedCard.className = "card";
      proposedCard.querySelectorAll("p span").forEach((el) => {
        el.classList.add("text-green-400");
      });
      baselineCard.querySelectorAll("p span").forEach((el) => {
        el.classList.add("text-red-400");
      });
    }

    document
      .getElementById("proposed-show-graph")
      .addEventListener("click", function () {
        const classLabels = Object.keys(data.proposed_probabilities);
        const proposedProbabilities = Object.values(
          data.proposed_probabilities
        );

        const classLegend = data["class_labels"];

        let canvas = document.createElement("canvas");
        canvas.id = "chartCanvas";

        const legendDiv = document.createElement("div");
        legendDiv.id = "legendDiv";
        legendDiv.style.marginTop = "20px";
        legendDiv.style.display = "grid";
        legendDiv.style.gridTemplateColumns =
          "repeat(auto-fill, minmax(150px, 1fr))";
        legendDiv.style.gap = "10px";
        legendDiv.style.fontSize = "12px";
        legendDiv.innerHTML = classLegend
          .map(
            (label, index) =>
              `<div style="text-align: start;"><strong>${index}:</strong> ${label}</div>`
          )
          .join("");

        Swal.fire({
          title: "Class Probabilities",
          html: `${canvas.outerHTML}${legendDiv.outerHTML}`,
          showConfirmButton: true,
          showCancelButton: true,
          confirmButtonText: '<span style="color: white;">Print</span>',
          confirmButtonColor: "#007bff", // Blue color for Print button
          cancelButtonText: "Close",
          width: 800,
          didOpen: () => {
            const ctx = document.getElementById("chartCanvas").getContext("2d");
            canvas = ctx;
            new Chart(ctx, {
              type: "bar",
              data: {
                labels: classLabels,
                datasets: [
                  {
                    label: "Probability (%)",
                    data: proposedProbabilities.map((p) =>
                      (p * 100).toFixed(2)
                    ),
                    backgroundColor: "rgba(75, 192, 192, 0.6)",
                    borderColor: "rgba(75, 192, 192, 1)",
                    borderWidth: 1,
                  },
                ],
              },
              options: {
                responsive: true,
                plugins: {
                  legend: {
                    display: true,
                    position: "top",
                  },
                },
                scales: {
                  y: {
                    beginAtZero: true,
                    max: 100,
                  },
                },
              },
            });
          },
        }).then((result) => {
          if (result.isConfirmed) {
            // Open new window for printing
            const printWindow = window.open(
              "",
              "PRINT",
              "width=800,height=600"
            );
            printWindow.document.write(
              "<html><head><title>Print Chart</title></head><body>"
            );
            printWindow.document.write("<h3>Class Probabilities</h3>");
            printWindow.document.write(canvas.outerHTML); // Add the chart
            printWindow.document.write(legendDiv.outerHTML); // Add the legend
            printWindow.document.write("</body></html>");
            printWindow.document.close();
            printWindow.focus();
            setTimeout(() => {
              printWindow.print();
              printWindow.close();
            }, 500); // Delay to ensure content is loaded
          }
        });
      });
  }
});

// document
//   .getElementById("proposed-show-graph")
//   .addEventListener("click", function () {
//     const canvas = document.createElement("canvas");
//     canvas.id = "chartCanvas";

//     Swal.fire({
//       title: "Class Probabilities",
//       html: canvas.outerHTML,
//       showConfirmButton: true,
//       didOpen: () => {
//         const ctx = document.getElementById("chartCanvas").getContext("2d");
//         new Chart(ctx, {
//           type: "bar",
//           data: {
//             labels: classLabels,
//             datasets: [
//               {
//                 label: "Probability",
//                 data: proposedProbabilities,
//                 backgroundColor: "rgba(75, 192, 192, 0.6)",
//                 borderColor: "rgba(75, 192, 192, 1)",
//                 borderWidth: 1,
//               },
//             ],
//           },
//           options: {
//             scales: {
//               y: {
//                 beginAtZero: true,
//               },
//             },
//           },
//         });
//       },
//     });
//   });

// document.addEventListener("DOMContentLoaded", function () {
//   // Get the canvas element
//   const ctx = document.getElementById("").getContext("2d");

//   // Example data (replace this with your dynamic data)
//   const speciesLabels = [
//     "Species A",
//     "Species B",
//     "Species C",
//     "Species D",
//     "Species E",
//   ];
//   const matchScores = [100, 75, 65, 55, 50]; // Match percentages

//   // Create the bar chart
//   const topSpeciesChart = new Chart(ctx, {
//     type: "bar",
//     data: {
//       labels: speciesLabels, // X-axis labels
//       datasets: [
//         {
//           label: "Match Percentage (%)",
//           data: matchScores, // Y-axis values
//           backgroundColor: [
//             "rgba(75, 192, 192, 0.6)",
//             "rgba(54, 162, 235, 0.6)",
//             "rgba(255, 206, 86, 0.6)",
//             "rgba(255, 99, 132, 0.6)",
//             "rgba(153, 102, 255, 0.6)",
//           ],
//           borderColor: [
//             "rgba(75, 192, 192, 1)",
//             "rgba(54, 162, 235, 1)",
//             "rgba(255, 206, 86, 1)",
//             "rgba(255, 99, 132, 1)",
//             "rgba(153, 102, 255, 1)",
//           ],
//           borderWidth: 1,
//           barThickness: 40,
//           maxBarThickness: 60,
//         },
//       ],
//     },
//     options: {
//       scales: {
//         y: {
//           beginAtZero: true,
//           title: {
//             display: true,
//             text: "Match Percentage (%)",
//           },
//         },
//         x: {
//           title: {
//             display: true,
//             text: "Species",
//           },
//         },
//       },
//       responsive: true,
//       plugins: {
//         legend: {
//           display: true,
//           position: "top",
//         },
//         tooltip: {
//           enabled: true,
//         },
//       },
//     },
//   });
// });
