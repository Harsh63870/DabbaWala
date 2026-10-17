# 🍱 DabbaETA — 2-Minute Demo Walkthrough Script

Welcome to **DabbaETA**, an open-source, local-first delivery ETA prediction and dispatch orchestration assistant built for small Indian tiffin services, home bakeries, cloud kitchens, and neighborhood couriers.

---

## ⏱️ Video / Live Walkthrough (2 Minutes)

### Minute 0:00 – 0:30 | Setup & Launch
1. **Launch the application**:
   ```bash
   # Terminal 1: Dev Server
   make dev
   # Server starts at http://localhost:8000
   ```
2. Open your browser to **`http://localhost:8000`**.
3. **Point out the core value proposition**:
   - **Privacy badge**: *"🔒 100% Local Customer Privacy"*. Customer names, phone numbers, and home addresses never leave the host laptop.
   - **Dual ML/LLM Stack**: Powered by TabPFN for tabular regression and Gemma (via Ollama) for autonomous tool routing.

---

### Minute 0:30 – 1:00 | Live Orders Queue & TabPFN ETA Prediction
1. Navigate to the **"📋 Orders & Risk Queue"** tab.
2. Observe the live orders:
   - Notice the **TabPFN ETA** column calculated in real-time.
   - Look at the **Delay Risk Badge**: `HIGH (82%)`, `MODERATE (54%)`, or `LOW (12%)`.
   - Toggle the predictor dropdown in the top right from **TabPFN** to **XGBoost** or **Linear Regression** to demonstrate multi-model support.
3. Test search filter: type `Bandra` or `Rainy` in the search box to filter orders by destination corridor or weather.

---

### Minute 1:00 – 1:30 | Gemma Agent & Chat in Hinglish
1. Switch to the **"💬 Dispatch Chat (Gemma)"** tab.
2. In the chat box, type or click the quick prompt:
   > **`Aaj kaunsi orders late hongi?`**
3. Notice the agent’s response:
   - Autonomous JSON tool routing triggers `list_risky_orders({"threshold": 0.45})`.
   - The tool badge displays: `⚡ Executed Tool: list_risky_orders`.
   - The response provides an executive list of delayed tiffins with customer names, promised windows, and delay percentages.
4. Try a message drafting prompt:
   > **`Draft a polite WhatsApp delay apology for Priya in Hinglish`**
   - The agent calls `draft_message`.
   - An authentic, warm apology message is generated:
     *"Namaste Priya ji! 🙏 Aapka dabba order (DB-10042) dispatch ho chuka hai. Heavy traffic ki wajah se lagbhag 15 minutes ka delay ho sakta hai..."*

---

### Minute 1:30 – 2:00 | Temporal Orchestration & Benchmark Science
1. In the top navigation bar, click the glowing green button:
   > **`⚡ Run Dispatch Workflow`**
2. The **Temporal Dispatch Modal** opens:
   - Watch the animated 5-step activity pipeline:
     1. `load_orders` (fetching pending shipments)
     2. `predict_all` (TabPFN batch inference with Sentry tracing)
     3. `flag_risky` (evaluating breach threshold)
     4. `explain` (Gemma root cause bottleneck diagnosis)
     5. `draft_messages` (generating batch WhatsApp notifications)
   - Read the **Logistics Bottleneck Analysis** diagnosed by Gemma.
   - View the drafted customer messages with the **"📋 Copy"** button and direct **"📲 Open WhatsApp"** (`https://wa.me`) link.
3. Close the modal and click the **"📊 Benchmark & Science"** tab:
   - Point out the side-by-side comparison table (TabPFN: **3.60 min MAE**, R² **0.937** vs Linear Regression: **6.27 min MAE**).
   - Show the generated plot (`docs/benchmark.png`) comparing performance across 50, 100, and 200 small-data training regimes.
4. **Wrap-up**: Local-first, zero API costs, resilient orchestration with Temporal, and private customer data.
