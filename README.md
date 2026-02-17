# NSE Screener 🚀

A Python-based stock screening tool for the **National Stock Exchange (NSE) of India**.

This project helps filter and analyze NSE-listed stocks using custom screening logic. It includes both a script-based screener and a lightweight web interface for viewing results interactively.

---

## 🧠 Overview

`nse_screrner` is designed for traders and investors who want to:

- Screen NSE stocks using defined rules  
- Generate ranked stock lists (e.g., Top 10)  
- Export results to Excel  
- View results in a simple web interface  

The project is structured to be easy to modify and extend with your own trading logic.

---

## 📂 Project Structure

```text
.
├── swing_screener.py          # Core stock screening logic
├── web_app.py                 # Web interface to display screening results
├── requirements.txt           # Python dependencies
├── Predicta_Top10.xlsx        # Example output file
└── __pycache__/               # Python cache files
```

---

## ⚙️ Features

- 📊 Custom stock screening logic  
- 🏆 Generates ranked stock lists (Top picks)  
- 📁 Excel export support  
- 🌐 Simple local web dashboard  
- 🧩 Easy to modify and extend  

---

## 🛠️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/brrrezy/nse_screrner.git
cd nse_screrner
```

### 2️⃣ Create a Virtual Environment (Recommended)

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🧾 Run the Screener Script

```bash
python swing_screener.py
```

This will execute the stock screening logic and may generate an Excel output file (e.g., `Predicta_Top10.xlsx`).

### 🌍 Run the Web App

```bash
python web_app.py
```

Then open your browser and go to:

```text
http://localhost:5000
```

You can view screened results through the web interface.

---

## 📈 Output

The project includes a sample output file:

```text
Predicta_Top10.xlsx
```

This file demonstrates how the screened stock results may be structured.

---

## 📦 Dependencies

All required Python packages are listed in:

```text
requirements.txt
```

Install them before running the project.

---

## 🧩 Customization

You can modify:

- Screening rules inside `swing_screener.py`
- UI layout inside `web_app.py`
- Ranking logic
- Output format

This project is meant to be flexible for swing traders and quantitative strategy builders.

---

## 🤝 Contributing

1. Fork the repository  
2. Create a feature branch  
3. Commit your changes  
4. Push to your branch  
5. Open a Pull Request  

---

## 📝 License

Copyright (c) 2026 Shivanshu Srivastav  
All rights reserved.

Unauthorized copying, modification, distribution, or use of this software is strictly prohibited without explicit written permission.

---

## ⚠️ Disclaimer

This project is for educational purposes only.  
It does not constitute financial advice. Always conduct your own research before making investment decisions.

---

Happy Building & Smart Screening 📊📈
