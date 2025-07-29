# CineMint: Telugu Box Office Revenue Predictor

**CineMint** is a machine learning-powered tool that predicts the worldwide box office revenue of Telugu movies before release. Combining historical data, feature-rich ML models, a simple API, and a user-friendly dashboard, CineMint helps producers, investors, and cinephiles estimate and analyze Telugu film performance with confidence.

---

## 🚀 Features

- **Fast Movie Revenue Prediction** (API/GUI)
- **Historical Data**: 100+ Telugu movies’ budgets, revenues, genres
- **Multiple Models**: Linear Regression, XGBoost (R² ≈ 0.85)
- **Live Dashboard**: Interactive, intuitive Streamlit front-end

---

## 📤 Sample API Usage

curl -X POST http://localhost:5000/predict
-H "Content-Type: application/json"
-d '{"title":"Test Movie","budget":80,"opening_theatres":2500,"opening_revenue":25,"genres":"Action|Drama","MPAA":"UA","release_year":2025,"release_month":8,"release_days":0}'

---

## 👩‍💻 License

MIT License
Copyright (c) 2025 MIT Manipal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


---

**Predict box office success, strategize your next blockbuster, and explore the future of Telugu cinema with CineMint!**
