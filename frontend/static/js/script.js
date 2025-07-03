document.addEventListener("DOMContentLoaded", function () {
  const sections = {
    upload: {
      element: document.getElementById("upload-section"),
      progress: 1,
      active: true,
    },
    results: {
      element: document.getElementById("result-section"),
      progress: 2,
      active: false,
    },
    save: {
      element: document.getElementById("save-section"),
      progress: 3,
      active: false,
    },
  };

  let currentSection = "upload";
  let currentChart = null;
  let currentData = null;
  let tempData = null;

  const progressButtons = document.querySelectorAll(".progress-btn");
  const fileInput = document.getElementById("input-file");
  const uploadButton = document.getElementById("upload-btn");
  const loader = document.getElementById("loader");
  const contentSection = document.getElementById("content-section");

  uploadButton.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", handleFileUpload);

  init();

  function init() {
    updateProgress(1);
  }
  progressButtons.forEach((button) => {
    button.addEventListener("click", async () => {
      const sectionId = button.getAttribute("data-section");

      if (!currentData) {
        Swal.fire({
          title: "No Image Uploaded",
          text: "Upload an image first to proceed.",
          icon: "warning",
          showCancelButton: true,
          showConfirmButton: false,
          cancelButtonText: '<span style="color: white;">Go back</span>',
          cancelButtonColor: "var(--color-accent)",
        });
        return;
      }

      if (sectionId === "upload" && sections.results.active) {
        const result = await Swal.fire({
          title: "Discard Progress?",
          text: "Are you sure you want to go back? Your results will be discarded.",
          icon: "warning",
          showCancelButton: true,
          confirmButtonText:
            '<span style="color: #cc2020;">Yes, go back</span>',
          confirmButtonColor: "#FFFFFF",
          cancelButtonText: '<span style="color: white;">Cancel</span>',
          cancelButtonColor: "var(--color-accent)",
        });

        if (!result.isConfirmed) return;

        sections.results.active = false;
      }

      if (sectionId !== currentSection) {
        updateProgress(sections[sectionId].progress);
      }
    });
  });

  function updateProgress(step) {
    // STEP 0 = LOADING

    if (step === 0) {
      setTimeout(() => {
        toggleVisibility(loader, true);
        toggleVisibility(contentSection, false);
      }, 200);
      return;
    }

    toggleVisibility(loader, false);
    toggleVisibility(contentSection, true);

    if (step === 1) {
      currentData = null;
      fileInput.value = "";
    }

    currentSection = Object.keys(sections).find(
      (key) => sections[key].progress === step
    );

    // UPDATE SECTION ACTIVE STATE
    Object.values(sections).forEach((section) => {
      if (section.progress <= step) {
        section.active = true;
      } else {
        section.active = false;
      }
      toggleVisibility(section.element, section.progress === step);
    });

    console.log("Section State:", sections);

    highlightProgress();
  }

  async function handleFileUpload() {
    if (fileInput.files.length > 0) {
      const file = fileInput.files[0];
      const formData = new FormData();
      formData.append("file", file);

      if (!file.type.startsWith("image/")) {
        Swal.fire({
          icon: "error",
          title: "Invalid File",
          text: "Please upload an image file.",
        });
        return;
      }

      if (!file.type.startsWith("image/")) {
        Swal.fire("Invalid File", "Please upload a valid image file.", "error");
        resetFileDisplay();
        return;
      }

      try {
        updateProgress(0);

        let response = await fetch("/predict", {
          method: "POST",
          body: formData,
        });

        currentData = await response.json();
        console.log("Response:", currentData);

        if (!currentData) {
          Swal.fire("Error", "Failed to process the file.", "error");
          resetFileDisplay();
          console.log("trigger error");
          return;
        }
        const models = Object.entries(currentData.models);
        const highlight = models.reduce((prev, curr) =>
          prev[1].confidence >= curr[1].confidence ? prev : curr
        )[0];

        models.forEach(([modelName, modelData]) => {
          displayResult(modelName, modelData, modelName === highlight);
        });
        document.querySelector("#result-section").classList.remove("hidden");

        const [dense_response, inception_response, mobilenet_response] =
          await Promise.all([
            // fetch("predict/model/vgg", { method: "POST", body: formData }),
            fetch("predict/model/densenet", { method: "POST", body: formData }),
            fetch("predict/model/inception", {
              method: "POST",
              body: formData,
            }),
            fetch("predict/model/mobilenet", {
              method: "POST",
              body: formData,
            }),
          ]);

        // Parse JSON responses concurrently
        const [dense_data, inception_data, mobilenet_data] = await Promise.all([
          // vgg_response.json(),
          dense_response.json(),
          inception_response.json(),
          mobilenet_response.json(),
        ]);

        // displayResult("vgg", vgg_data, true);
        displayResult("dense", dense_data, true);
        displayResult("inception", inception_data, true);
        displayResult("mobilenet", mobilenet_data, true);

        updateProgress(2);
      } catch (error) {
        console.error("Error:", error);
        Swal.fire("Error", "An error occurred while uploading.", "error");
        updateProgress(1);
      }
    } else {
      updateProgress(1);
    }
  }

  function highlightProgress() {
    progressButtons.forEach((button) => {
      const sectionId = button.getAttribute("data-section");
      const section = sections[sectionId];

      if (section.active) {
        button.classList.add("bg-[var(--color-accent)]", "text-white");
        button.classList.remove(
          "bg-[var(--color-gray)]",
          "text-[var(--color-secondary)]"
        );
      } else {
        button.classList.remove("bg-[var(--color-accent)]", "text-white");
        button.classList.add(
          "bg-[var(--color-gray)]",
          "text-[var(--color-secondary)]"
        );
      }
    });

    Object.keys(sections).forEach((key) => {
      const section = sections[key];
      const button = Array.from(progressButtons).find(
        (btn) => btn.getAttribute("data-section") === key
      );
      const indicator = button.closest(".progress-indicator");
      const prevLine = indicator.previousElementSibling;

      // Ensure the active class is added first
      if (section.active) {
        indicator.classList.add("active");
      } else {
        indicator.classList.remove("active");
      }

      // Highlight the previous line for all completed sections
      if (prevLine) {
        if (section.progress <= sections[currentSection].progress) {
          prevLine.style.backgroundColor = "var(--color-accent)";
        } else {
          prevLine.style.backgroundColor = "var(--color-gray)";
        }
      }
    });
  }

  function toggleVisibility(element, show, displayClass = "block") {
    element.classList.toggle("hidden", !show);
    element.classList.toggle(displayClass, show);
  }
  function displayResult(name, data, highlight = false) {
    console.log("Displaying Result:", data);
    // Prediction
    document
      .querySelectorAll(`.${name}-pred`)
      .forEach((el) => (el.textContent = data.prediction));
    // Confidence
    document.querySelectorAll(`.${name}-conf`).forEach((el) => {
      el.textContent = `${(data.confidence * 100).toFixed(2)}`;
    });
    // Prediction Time
    document.querySelectorAll(`.${name}-speed`).forEach((el) => {
      el.textContent = `${data.prediction_time.toFixed(2)}`;
    });

    // Set images
    const modelImg = document.querySelector(`#${name}-card img`);
    modelImg.src = data.image;

    // Set card by highest confidence
    const modelCard = document.getElementById(`${name}-card`);
    if (highlight) {
      modelCard.className = "card";
    } else {
      modelCard.className = "card-plain";
    }

    const graphListener = function () {
      init_swal(data.probabilities, currentData.class_labels);
    };

    document
      .getElementById(`${name}-show-graph`)
      .removeEventListener("click", graphListener);

    document
      .getElementById(`${name}-show-graph`)
      .addEventListener("click", graphListener);
  }

  function init_swal(probabilities, class_labels) {
    const classLabels = Object.keys(probabilities);
    const class_probabilities = Object.values(probabilities);

    const classLegend = class_labels;

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
      confirmButtonColor: "#007bff",
      cancelButtonText: "Close",
      width: 800,
      didOpen: () => {
        const ctx = document.getElementById("chartCanvas").getContext("2d");

        if (currentChart) {
          currentChart.destroy();
        }

        currentChart = new Chart(ctx, {
          type: "bar",
          data: {
            labels: classLabels,
            datasets: [
              {
                label: "Probability (%)",
                data: class_probabilities.map((p) => (p * 100).toFixed(2)),
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
            animation: {
              onComplete: () => {
                // Convert the fully rendered canvas to an image
                const imgData = ctx.canvas.toDataURL("image/png");
                document
                  .getElementById("chartCanvas")
                  .setAttribute("data-img", imgData);
              },
            },
          },
        });
        const confirmButton = Swal.getConfirmButton();
        confirmButton.disabled = true;

        setTimeout(() => {
          confirmButton.disabled = false;
        }, 3000);
      },
    }).then((result) => {
      if (result.isConfirmed) {
        const imgData = document
          .getElementById("chartCanvas")
          .getAttribute("data-img");
        const printWindow = window.open("", "PRINT", "width=800,height=600");
        printWindow.document.write(
          "<html><head><title>Print Chart</title></head><body>"
        );
        printWindow.document.write("<h3>Class Probabilities</h3>");
        printWindow.document.write(
          `<img src="${imgData}" style="width: 100%;"/><br>`
        );
        printWindow.document.write(legendDiv.outerHTML);
        printWindow.document.write("</body></html>");
        printWindow.document.close();
        printWindow.focus();
        setTimeout(() => {
          printWindow.print();
          printWindow.close();
        }, 500);
      }
    });
  }
});
