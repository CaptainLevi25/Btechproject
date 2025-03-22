const dropArea = document.getElementById("dropArea");
const fileInput = document.getElementById("fileInput");
const imageContainer = document.getElementById("imageContainer");
const previewImage = document.getElementById("previewImage");
const predictBtn = document.getElementById("predictBtn");
const banner = document.getElementById("banner");
const bannerText = document.getElementById("bannerText");
const resultContainer = document.getElementById("resultContainer");
const resultText = document.getElementById("resultText");

function getTranslationValues(element) {
  const style = window.getComputedStyle(element);
  const matrix = new DOMMatrixReadOnly(style.transform);

  return {
    translateX: matrix.m41, // X translation
    translateY: matrix.m42, // Y translation
  };
}

function translateElement(element, deltaX, deltaY) {
  // Get current translation values
  const { translateX, translateY } = getTranslationValues(element);

  // Apply new transformation from current position
  element.style.transform = `translate(${translateX + deltaX}px, ${
    translateY + deltaY
  }px)`;
  
}
function isTranslated(element) {
  const style = window.getComputedStyle(element);
  return style.transform !== "none";
}

// Drag & Drop Effects
dropArea.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropArea.classList.add("dragover");
});

dropArea.addEventListener("dragleave", () => {
  dropArea.classList.remove("dragover");
});

dropArea.addEventListener("drop", (event) => {
  event.preventDefault();
  dropArea.classList.remove("dragover");
  const files = event.dataTransfer.files;
  if (files.length > 0) {
    handleFile(files[0]);
  }
});

fileInput.addEventListener("change", (event) => {
  if (event.target.files.length > 0) {
    handleFile(event.target.files[0]);
  }
});

function handleFile(file) {
  if (isTranslated(dropArea) && file && file.type.startsWith("image/")) {
 
    const reader = new FileReader();
    reader.onload = function (e) {
      previewImage.src = e.target.result;
      imageContainer.style.display = "block";
      setTimeout(() => {
        imageContainer.style.opacity = "1";
        predictBtn.classList.add("show");
        //predictBtn.style.transform  =  "translateY(90px)";
        translateElement(predictBtn, 0, 90);
        if (imageContainer.style.opacity !== "0") {
          dropArea.style.opacity = "0";
        } else {
          dropArea.style.opacity = "1";
        }
      }, 200);
    };
    reader.readAsDataURL(file);
  } else if (file && file.type.startsWith("image/")) {
   
    const reader = new FileReader();
    reader.onload = function (e) {
      previewImage.src = e.target.result;
      imageContainer.style.display = "block";
      setTimeout(() => {
        imageContainer.style.opacity = "1";
        predictBtn.classList.add("show");
      }, 200);
    };
    reader.readAsDataURL(file);
  }
}

// Predict Button Click Handler
predictBtn.addEventListener("click", async () => {
  if (!previewImage.src) {
    alert("Please upload an image first!");
    return;
  }

  // Show loading effect
  predictBtn.classList.add("loading");
  predictBtn.innerHTML = ""; // Remove text
  predictBtn.disabled = true;

  // Simulate API Call (Replace with actual API)
  setTimeout(async () => {
    try {
      // Simulated API response
      const response = await fakeApiCall();
      resultText.innerHTML = response;

      // Animate Banner (Move to Top & Shrink)
      //   banner.style.transform = "translateY(-199px) ";
      //   banner.style.transform = "translateX(199px)";
      banner.style.transform = "translate(0px , -220px)";
      banner.style.fontSize = "18px";
        banner.style.width = "990px";
      //bannerText.style.width = "100%";
      bannerText.innerHTML =
        "Your AI-Powered Personalized Skin Cancer Detector";

      // Move Upload & Predict Container to Right
      imageContainer.style.zIndex = "-1";
      imageContainer.style.opacity = "0";

      if (imageContainer.style.opacity !== "0") {
        dropArea.style.opacity = "0";
      } else {
        dropArea.style.opacity = "1";
      }

      dropArea.style.transform = "translateX(-700px)";

      predictBtn.style.transform = "translateX(-700px)";

      // Show Result Container with Animation
      resultContainer.style.display = "block";
      setTimeout(() => {
        resultContainer.style.opacity = "1";
      }, 300);
    } catch (error) {
      alert("Error fetching result. Try again.");
    } finally {
      // Reset Predict Button
      predictBtn.classList.remove("loading");
      predictBtn.innerHTML = "Predict";
      predictBtn.disabled = false;
    }
  }, 2000); // Simulating a 2-second API response time
});

function fakeApiCall() {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(
        "<b>The image shows signs of Pyogenic granulomas and hemorrhage (Can lead to cancer).</b>"
      );
    }, 1000);
  });
}
