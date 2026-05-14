from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import tldextract
import os

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Running on: {device} ---")

# ================= 1. VERIFIED DOMAIN DATABASE =================
# Only the REGISTERED DOMAIN is checked — not subdomain
# So infosys.scam.com → domain is "scam" → NOT verified → flagged
# But infosys.com → domain is "infosys" → verified → legitimate
VERIFIED_DOMAINS = {
    "google", "infosys", "onwingspan", "claude", "github",
    "microsoft", "apple", "amazon", "facebook", "instagram",
    "wikipedia", "youtube", "linkedin", "twitter", "x", "openai",
    "medium", "unsplash", "behance", "dribbble", "anthropic",
    "netflix", "spotify", "adobe", "zoom", "slack", "notion",
    "figma", "canva", "stackoverflow", "reddit", "quora",
    "yahoo", "bing", "paypal", "stripe", "shopify", "wordpress",
    "wix", "squarespace", "godaddy", "cloudflare", "vercel",
    "heroku", "aws", "azure", "dropbox", "box", "atlassian",
    "jira", "confluence", "hubspot", "salesforce", "oracle",
    "ibm", "cisco", "intel", "nvidia", "amd", "samsung",
    "sony", "lg", "dell", "hp", "lenovo", "asus", "tata",
    "wipro", "hcl", "tcs", "cognizant", "accenture"
}

# ================= 2. LOAD URL MODEL (HuggingFace BERT) =================
print("Loading BERT URL Model...")
url_classifier = pipeline(
    "text-classification",
    model="ealvaradob/bert-finetuned-phishing",
    device=0 if torch.cuda.is_available() else -1
)
print("URL Model ready.")

# ================= 3. LOAD IMAGE MODEL (EfficientNet-B3) =================
print("Loading EfficientNet-B3 Image Model...")
image_model = models.efficientnet_b3(weights=None)
in_features = image_model.classifier[1].in_features
image_model.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features, 2)
)

try:
    image_model.load_state_dict(torch.load("image_model_efficientnet.pth", map_location=device))
    print("Image weights loaded successfully.")
except Exception as e:
    print(f"WARNING: Could not load image weights: {e}")
    print("Image model will run with random weights (low accuracy until retrained)")

image_model = image_model.to(device).eval()

image_transform = transforms.Compose([
    transforms.Resize((300, 300)),   # EfficientNet-B3 native size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# image_classes[0] = legitimate, image_classes[1] = phishing
# Must match the folder order during your training (alphabetical = legitimate first)
image_classes = ["legitimate", "phishing"]

# ================= 4. URL ANALYSIS LOGIC =================
def analyze_url(url: str):
    """
    3-layer URL check:
    1. Verified Domain  → auto legitimate
    2. Impersonation    → auto phishing
    3. BERT AI fallback → for unknown sites
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    ext = tldextract.extract(url)
    actual_domain = ext.domain.lower()   # e.g. "infosys" from "infosys.com"
    subdomain     = ext.subdomain.lower() # e.g. "infosys" from "infosys.scam.com"

    # Layer 1: Verified registered domain
    if actual_domain in VERIFIED_DOMAINS:
        return "legitimate", 1.0, f"Verified Domain: {actual_domain.upper()}"

    # Layer 2: Brand used in subdomain of unverified domain (impersonation)
    for brand in VERIFIED_DOMAINS:
        if brand in subdomain:
            return "phishing", 1.0, f"Impersonation Detected: '{brand}' in untrusted domain"

    # Layer 3: BERT AI pattern analysis
    try:
        result = url_classifier(url, truncation=True, max_length=512)[0]
        raw_label = result["label"].lower()
        conf      = result["score"]
        # ealvaradob model: LABEL_1 = phishing, LABEL_0 = legitimate
        label = "phishing" if "1" in raw_label or "phish" in raw_label else "legitimate"
        return label, conf, "AI URL Analysis"
    except Exception as e:
        print(f"URL model error: {e}")
        return "legitimate", 0.5, "URL Analysis Error"

# ================= 5. MAIN ROUTE =================
@app.route("/", methods=["GET", "POST"])
def scan():
    if request.method == "GET":
        return render_template("index.html")

    url        = request.form.get("url", "").strip()
    image_file = request.files.get("image")

    if not url or not image_file:
        return jsonify({"error": "Missing URL or image"}), 400

    print(f"\n{'='*50}")
    print(f"[SCAN] {url}")

    # --- STEP 1: URL ANALYSIS ---
    url_label, url_conf, url_source = analyze_url(url)
    print(f"URL  → {url_label} ({url_conf:.4f}) | {url_source}")

    # --- STEP 2: IMAGE ANALYSIS (always runs) ---
    os.makedirs("uploads", exist_ok=True)
    img_path = os.path.join("uploads", "last_scan.png")
    image_file.save(img_path)

    print("Running EfficientNet-B3 on screenshot...")
    img = Image.open(img_path).convert("RGB")
    img_tensor = image_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs   = image_model(img_tensor)
        probs     = torch.softmax(outputs, dim=1)[0]
        img_idx   = torch.argmax(probs).item()
        img_conf  = float(probs[img_idx].item())
        img_label = image_classes[img_idx]

    print(f"IMAGE → {img_label} ({img_conf:.4f})")

    # --- STEP 3: HYBRID FINAL DECISION ---
    if "Verified Domain" in url_source:
        # Trusted brand — URL wins but still show image result
        final_label  = "legitimate"
        final_conf   = url_conf
        final_source = url_source

    elif "Impersonation" in url_source:
        # Hard veto — impersonation detected
        final_label  = "phishing"
        final_conf   = url_conf
        final_source = url_source

    elif url_label == "phishing" and url_conf >= 0.98:
        # URL model is very confident it's phishing — URL wins
        final_label  = "phishing"
        final_conf   = url_conf
        final_source = f"URL Veto ({url_source})"

    elif img_conf > url_conf:
        # Image model is more confident
        final_label  = img_label
        final_conf   = img_conf
        final_source = "Image Model"

    else:
        # URL model is more confident
        final_label  = url_label
        final_conf   = url_conf
        final_source = "URL Model"

    print(f"FINAL → {final_label} ({final_conf:.4f}) | {final_source}")
    print(f"{'='*50}\n")

    return jsonify({
        "url_label":   url_label,
        "url_conf":    round(float(url_conf), 4),
        "img_label":   img_label,
        "img_conf":    round(float(img_conf), 4),
        "final_label": final_label,
        "final_conf":  round(float(final_conf), 4),
        "source":      final_source,
    })


if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=True)