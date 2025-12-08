AI Financial Advisor aims to leverage Agentic AI to analyze personal financial data and generate a tailored financial advice report that empowers con-sumers to better understand their finances and make strategic decisions.  It helps users reduce debt, optimize savings, plan for major life goals, and invest with confidence, all while ensuring compliance with financial regulations. The AI fi-nancial advisor will help to analyze personal financial data, cluster them into customer segments based on their financial behavior, retrieve market data, create a plan for asset allocation, validate the advice against regulatory compliance, and generate a personalized financial advice report. The Financial advice report will have the consumer’s current financial health summary based on the customer segment, recommendations on their financial be-havior, and portfolio recommendations in compli-ance with regulatory data. Optionally, the report would also contain educational content focusing on financial literacy for the consumer.

The project leverages the following components to generate financial advice:  

1. Machine Learning Clustering Algorithm - to segment customer's financial behavior.  
2. Deep learning Financial Advisor model - train the Model with financial terminology, concepts, and relationships.  
3. Retrieval augmented reality that leverages Financial advisor model and external knowledge source to generate financial planning and asset allocation recommendation
4.  Agentic AI pipeline to wire all the components and geenrate a consolidated advice.

Github Files

1. AAI_590_KarthikRaghavan.ipynb - Main notebook that contains the code for fin advisor
2. finadvisor_finetune.ipynb - Code for deep learning model
3. data/docs - External knowledge docs used for RAG
4. data/userdata - user input data for evaluation
5. data/reports - AI Respose for different user input data
6. data/prompts - Prompts used for RAG
7. faiss_vector_store - Vector store db
8. results - KMeans clustering models

   

Data Summary

1. Personal Finance Tracker Dataset
2. Financial Glossary Dataset
3. External knowledge source for Financial planning advice and Portfolio asset allocation.
4. Ground truth for RAG Evaluation

Model Architecture

Agentic Pipeline

<img width="2010" height="856" alt="image" src="https://github.com/user-attachments/assets/02bf8e04-83d0-472d-937c-eb27a84beaef" />

Deep learning Model & RAG Pipeline

<img width="1994" height="844" alt="image" src="https://github.com/user-attachments/assets/bde964f1-f5e6-43ea-baed-ff911a47901d" />

Model Training

<img width="1822" height="920" alt="image" src="https://github.com/user-attachments/assets/e2791bb5-d5aa-4fca-a497-628441c8bc8f" />

Sample Inference - Financial Planning

<img width="289" height="206" alt="image" src="https://github.com/user-attachments/assets/4b8af216-5587-42d8-928f-5a7c1042cf07" />

Sample Inference - Asset Allocation

<img width="310" height="186" alt="image" src="https://github.com/user-attachments/assets/41814733-b5b0-4ee8-95b8-27626fd53d1d" />

Model Evaluation 

<img width="660" height="822" alt="image" src="https://github.com/user-attachments/assets/993267e5-2f76-4d5e-9eb2-38e6d8f1b13f" />

<img width="1234" height="556" alt="image" src="https://github.com/user-attachments/assets/577054d7-4816-423d-a577-1731c4ee8c69" />

<img width="1234" height="458" alt="image" src="https://github.com/user-attachments/assets/a9a4c1bf-cae1-4662-87f5-6689075f24b4" />



Meaningful clusters enable differentiated advice.
RAG scores show strong grounding in expert guidelines.
Output reports were accurate, internally consistent, and aligned with real-world financial planning frameworks.
<img width="569" height="68" alt="image" src="https://github.com/user-attachments/assets/42979697-d788-41c1-abdb-79feac49d775" />















