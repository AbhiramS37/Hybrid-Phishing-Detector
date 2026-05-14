const FLASK_URL = "http://127.0.0.1:5000/";

let currentTab = null;

document.addEventListener("DOMContentLoaded", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  currentTab = tab;
  document.getElementById("currentUrl").textContent = tab.url;
});

document.getElementById("scanBtn").addEventListener("click", async () => {
  showLoader(true);
  hideResults();
  hideError();

  try {
    const url = currentTab.url;

    // Capture screenshot
    const screenshotDataUrl = await chrome.tabs.captureVisibleTab(null, { format: "png" });
    const imageBlob = dataURLToBlob(screenshotDataUrl);

    // Build FormData
    const formData = new FormData();
    formData.append("url", url);
    formData.append("image", imageBlob, "screenshot.png");

    // POST to Flask — header tells Flask to return JSON
    const response = await fetch(FLASK_URL, {
      method: "POST",
      headers: { "X-Requested-With": "extension" },
      body: formData,
    });

    if (!response.ok) throw new Error(`Server error: ${response.status}`);

    const data = await response.json();
    showResults(data);

  } catch (err) {
    showError(err.message);
  } finally {
    showLoader(false);
  }
});

function dataURLToBlob(dataUrl) {
  const [header, base64] = dataUrl.split(",");
  const mime = header.match(/:(.*?);/)[1];
  const binary = atob(base64);
  const arr = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) arr[i] = binary.charCodeAt(i);
  return new Blob([arr], { type: mime });
}

function isSafe(label) {
  return label.toLowerCase().includes("legit");
}

function confToPercent(val) {
  if (!val && val !== 0) return 0;
  return val <= 1 ? val * 100 : val;
}

function showResults(data) {
  const safe = isSafe(data.final_label);
  const cls  = safe ? "safe" : "phishing";

  // Badge
  const badge = document.getElementById("verdictBadge");
  badge.textContent = safe ? "Safe" : "Phishing";
  badge.className = `verdict-badge ${cls}`;

  // URL row
  const urlSafe = isSafe(data.url_label);
  document.getElementById("urlLabel").textContent = data.url_label;
  document.getElementById("urlResult").className = `rval ${urlSafe ? "safe" : "phishing"}`;
  document.getElementById("urlConf").textContent = (confToPercent(data.url_conf).toFixed(1)) + "%";
  const urlBar = document.getElementById("urlBar");
  urlBar.className = `conf-fill ${urlSafe ? "safe" : "phishing"}`;
  setTimeout(() => { urlBar.style.width = confToPercent(data.url_conf) + "%"; }, 100);

  // Image row
  const imgSafe = isSafe(data.img_label);
  document.getElementById("imgLabel").textContent = data.img_label;
  document.getElementById("imgResult").className = `rval ${imgSafe ? "safe" : "phishing"}`;
  document.getElementById("imgConf").textContent = (confToPercent(data.img_conf).toFixed(1)) + "%";
  const imgBar = document.getElementById("imgBar");
  imgBar.className = `conf-fill ${imgSafe ? "safe" : "phishing"}`;
  setTimeout(() => { imgBar.style.width = confToPercent(data.img_conf) + "%"; }, 150);

  // Source
  document.getElementById("sourceVal").textContent = data.source;

  // Final
  document.getElementById("finalLabel").textContent = data.final_label;
  document.getElementById("finalConf").textContent = (confToPercent(data.final_conf).toFixed(1)) + "%";
  document.getElementById("finalRow").className = `final-row ${cls}`;

  document.getElementById("results").classList.add("visible");
}

function showLoader(show) {
  document.getElementById("loader").classList.toggle("visible", show);
  document.getElementById("scanBtn").disabled = show;
}

function hideResults() {
  document.getElementById("results").classList.remove("visible");
}

function hideError() {
  const el = document.getElementById("errorBox");
  el.classList.remove("visible");
  el.textContent = "";
}

function showError(msg) {
  const el = document.getElementById("errorBox");
  el.textContent = "⚠ " + msg;
  el.classList.add("visible");
}