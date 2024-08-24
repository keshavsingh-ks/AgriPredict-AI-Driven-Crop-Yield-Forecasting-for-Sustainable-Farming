# AgriPredict: AI-Driven Crop Yield Forecasting for Sustainable Farming

## Overview

Agriculture forms the backbone of the Indian economy, employing over 50% of the workforce and contributing around 18% to the nation's GDP. However, the sector faces numerous challenges, from climate change to soil degradation, which can drastically impact crop yield and farmers' livelihoods. **AgriPredict** leverages the power of machine learning, cloud computing, and modern software engineering to predict crop yields accurately, optimize farming practices, and ensure sustainable agricultural growth.

## Key Features

- **Crop Yield Prediction**: Utilizing historical data, weather patterns, and soil characteristics, our model predicts crop yield and provides actionable insights for farmers.
- **Real-time Recommendations**: The system suggests optimal times for fertilization and other agricultural practices.
- **Web Interface**: A user-friendly web application powered by Flask allows farmers to interact with the model, input data, and receive predictions.

## Applications

- The application can be used by farmers to understand various parameters required to grow the right type of crop during the appropriate season.
- This project aims to reduce suicide rates among farmers due to lack of income.
- It is user-friendly and easy to use.
- It’s cost-effective and time-saving.
- Crop yield prediction is an essential task for decision-makers at national and regional levels (e.g., the EU level) for rapid decision-making.
- An accurate crop yield prediction model can help farmers decide what to grow and when to grow it.
- ML techniques used in this project will result in cost-effective solutions in the agricultural sector.

## Machine Learning Techniques

![Machine Learning Pipeline](./images/Picture3.png)
![Algorithm Comparison](./images/Picture5.png)

1. **Data Collection & Cleaning**
   - Collected extensive datasets including weather data, soil type, rainfall records, and historical crop yields.
   - Cleaned and pre-processed the data to remove inaccuracies and ensure high-quality inputs for the model.

2. **Model Training**
   - Implemented multiple machine learning algorithms, including Random Forest, Support Vector Machine (SVM), Decision Tree, and K-Nearest Neighbors (KNN).
   - The Random Forest model was chosen for its superior accuracy and robustness after thorough comparison with other algorithms.

3. **Model Deployment**
   - Deployed the trained model using a Flask web application, making it accessible through a user-friendly interface.
   - Implemented real-time prediction capability using live inputs from users.

4. **Results**
   - Achieved significant accuracy in crop yield predictions, enabling better decision-making for farmers.
   - Visualized the performance of various algorithms to select the most efficient one.

## AWS Integration

![AWS Architecture](./images/Picture1.png)

1. **AWS S3**: 
   - All datasets are securely stored in AWS S3, ensuring high availability and scalability.
   - Configured S3 bucket policies to manage access control and data security.

2. **AWS Lambda**:
   - Automated data processing tasks using AWS Lambda functions, reducing manual intervention and streamlining operations.
   - Lambda functions handle data retrieval, model execution, and result storage, making the system highly responsive and scalable.

3. **AWS EC2 & ECR**:
   - Deployed the Flask web application on an AWS EC2 instance, providing the necessary computational power to handle multiple user requests simultaneously.
   - Used AWS ECR to store Docker images of the model, ensuring quick recovery and consistent environment replication in case of any system failures.

## Docker Implementation

![Docker Architecture](./images/Picture2.png)

- **Dockerfile**: The project is containerized using Docker, which simplifies the deployment process and ensures consistency across different environments.
- **Container Orchestration**: Leveraged Docker to orchestrate multiple services, including the Flask application and the backend model, ensuring seamless communication and scaling.

## System Design

The system is designed to be modular, scalable, and efficient, with a clear separation of concerns between data handling, model execution, and user interaction.

- **Front-End**: Developed using HTML, CSS, and JavaScript, providing an intuitive interface for farmers to interact with the system.
- **Back-End**: The machine learning model is integrated into the Flask framework, with endpoints exposed for predictions and data management.
- **Data Flow**: Data is collected from the user, processed through the ML model, and the results are displayed on the web interface.

## Conclusion

**AgriPredict** is a robust, scalable, and user-friendly tool designed to assist farmers in making data-driven decisions to optimize crop yield. By integrating advanced machine learning techniques with the power of cloud computing, we provide a solution that not only predicts outcomes but also empowers users with actionable insights.
