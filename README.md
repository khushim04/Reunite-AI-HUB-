# Smart Lost & Found System

A web-based application to help users report and track **lost and found items** efficiently. This system integrates **AI/ML image-text matching**,
QR code scanning, and a MongoDB backend to make reporting and matching items seamless and accurate.

---

## Overview

Lost and found situations can be time-consuming and confusing. This system allows users to:  
- Report lost or found items with descriptions and images.  
- Match lost items with found items using AI-powered image-text similarity.  
- Scan QR codes to quickly access reports.  
- Maintain a centralized database for easy tracking.  

It is designed for organizations, campuses, or communities where item tracking is essential.

---

## Features

- Add **Lost Item Reports** with images and details.  
- Add **Found Item Reports** and track potential matches.  
- **AI/ML Integration** using CLIP/TensorFlow for image-text matching.  
- **QR Code Scanner** to quickly access reports.  
- Centralized storage and management using **MongoDB**.  

---

## Technologies Used

- **Frontend:** HTML, CSS, JavaScript  
- **Backend:** Python (Flask)  
- **Database:** MongoDB  
- **AI/ML:** TensorFlow / CLIP for image-text similarity  
- **Deployment:** Render / AWS  

---

## Make sure index.html should be in templater folder

## Installation

Follow these steps to run the project locally:

1. **Clone the repository:**
    
2. **Set up a virtual enviornment**
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

3. **Install dependencies:**
pip install -r requirements.txt

4. **Run the application:**
python app.py

5. **Access the app in your browser:**

