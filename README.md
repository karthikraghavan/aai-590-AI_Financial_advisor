AI Financial Advisor aims to leverage Agentic AI to analyze personal financial data and generate a tailored financial advice report that empowers con-sumers to better understand their finances and make strategic decisions.  It helps users reduce debt, optimize savings, plan for major life goals, and invest with confidence, all while ensuring compliance with financial regulations. The AI fi-nancial advisor will help to analyze personal financial data, cluster them into customer segments based on their financial behavior, retrieve market data, create a plan for asset allocation, validate the advice against regulatory compliance, and generate a personalized financial advice report. The Financial advice report will have the consumer’s current financial health summary based on the customer segment, recommendations on their financial be-havior, and portfolio recommendations in compli-ance with regulatory data. Optionally, the report would also contain educational content focusing on financial literacy for the consumer.

The project leverages the following components to generate financial advice:  

1. Machine Learning Clustering Algorithm - to segment customer's financial behavior.  
2. Deep learning Financial Advisor model - train the Model with financial terminology, concepts, and relationships.  
3. Retrieval augmented reality that leverages Financial advisor model and external knowledge source to generate financial planning and asset allocation recommendation
4.  Agentic AI pipeline to wire all the components and geenrate a consolidated advice.

Personal Finance Tracker Dataset 

Used for Financial Behavior Segmentation

Population: User’s personal finance data
# of Samples: 3000
# of features: 25
Missing data: None
Prediction Type: Clustering
# of Continuous features: 
# of Discrete features: 

The task is to segment based on the client’s financial behavior
<img width="308" height="177" alt="image" src="https://github.com/user-attachments/assets/d2483088-bb65-49b2-8e0d-9240b3ff6cb6" />


Financial Glossary Dataset

The FinRAD dataset consists of financial terms and their definitions.

Population: financial terms & definitions
# of samples: 1500
# of features: 12
Primary Features: “terms”, ”definitions.”

The task is to create a financial advisor model trained on financial glossary
<img width="312" height="141" alt="image" src="https://github.com/user-attachments/assets/f37c0121-37fd-40dc-879f-cf080a5d917b" />

External knowledge
 
Financial Planning: A Step-by-Step Guide

The task is to leverage this document as an external knowledge source in the Retrieval Augmented Reality pipeline to create financial advice.
<img width="584" height="81" alt="image" src="https://github.com/user-attachments/assets/e0d7be73-76d1-451a-8dc7-cd5e66eaac68" />

Portfolio Asset Allocation Strategies

The task is to leverage this document as an external knowledge source in the RAG pipeline to create a portfolio asset allocation recommendation.
<img width="602" height="45" alt="image" src="https://github.com/user-attachments/assets/451ae2d9-bfd6-4999-a110-e8218807d523" />

Ground Truth

Ground truth for evaluation

The ground truth is a manually curated CSV database, augmented with external knowledge documents from NERD Wallet. The database consists of queries, reference answers, ground truth, financial segments, and categories


<img width="924" height="81" alt="image" src="https://github.com/user-attachments/assets/47fb8e69-ce1a-4f70-8811-840511b70cdd" />




