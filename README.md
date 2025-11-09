# 🩺 PerfectHealth
**AI-Powered Health Insights** that detect trends in your WHOOP and lifestyle data — helping you understand *why* your body feels the way it does and how to recover faster.

---

## 🚀 Overview
PerfectHealth connects your physiological metrics (sleep, recovery, HRV, alcohol, caffeine, strain) with your lifestyle events to deliver **personalized AI recovery plans**.  
It leverages **AWS Lambda**, **Bedrock (Claude Sonnet 4.5)**, and **S3** to process your data, detect trends, and generate daily insights — all visualized on a modern web dashboard.

---

## 🧠 Inspiration
Wearables give us numbers, not meaning.  
PerfectHealth bridges the gap — turning your sleep and recovery data into clear, actionable insights like:  
> "You're on day 3 of low recovery — prioritize sleep and hydration to bounce back by Wednesday."

---

## 🧩 How It Works
1. **Data Collection:** A WHOOP-like dataset (stored in S3) with metrics like sleep, HRV, and recovery.
2. **AI Analysis:** AWS Lambda loads the data and queries Bedrock (Claude Sonnet 4.5) for AI-generated health insights.
3. **Visualization:** The frontend dashboard (HTML/CSS/JS) fetches the live insights through API Gateway and displays them beautifully.
4. **Optional Extensions:** Trendlines and historical comparisons (sleep efficiency, strain, temperature) can be toggled for deeper analysis.

---

## ⚙️ Tech Stack
| Layer | Technology |
|-------|-------------|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | AWS Lambda (Python), API Gateway |
| **AI Model** | Anthropic Claude Sonnet 4.5 via AWS Bedrock |
| **Storage** | Amazon S3 (JSON dataset) |
| **Visualization** | D3.js, Matplotlib |
| **Hosting (optional)** | GitHub Pages / Vercel |

---

## 💻 Run It Yourself

Clone the repo:
```bash
git clone https://github.com/vedantajwani/PerfectHealth.git
cd PerfectHealth/frontend
```

Then open `index.html` in your browser to view the dashboard.


