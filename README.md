# Medical Assistant Bot

## Problem Statement
This project aims to develop a medical question-answering system that responds to user queries regarding medical diseases using a provided dataset. The goal is to utilize natural language processing (NLP) techniques to effectively answer these queries.

## Approach

### 1. Data Preprocessing
The dataset was first processed to ensure proper structure for training. The following steps were taken:
- Tokenization of questions and contexts.
- Padding and truncation of inputs to ensure uniform length.
- Label creation for start and end positions of answers in the context.

### 2. Model Training
Three models were employed to improve the quality of answers:
- **all-MiniLM-L6-v2** for sentence embedding. It is a small, efficient transformer model designed for sentence embeddings. It's part of the MiniLM family, which focuses on providing lightweight models for NLP tasks. 
- **BERT (BioBERT)** was used for understanding and extracting relevant information from the provided context.
  - BioBERT (Bidirectional Encoder Representations from Transformers for Biomedical Text Mining) is a variant of the BERT model pre-trained specifically on large-scale biomedical corpora. It was developed to improve performance on tasks related to biomedical text processing
- **GPT-2** was utilized to generate more human-like, fluent answers based on BERT’s output.

#### Fine-Tuning
The model was fine-tuned with **80% of the data** for training and **20% for validation**. This split was chosen to maximize model accuracy while ensuring that the model generalizes well and avoids overfitting. Early stopping was also applied during training based on validation loss to prevent overtraining.

**Assumptions:**
- Questions in the dataset are representative of real-world medical queries.
- The context provided in the dataset contains sufficient information for answering the questions.
- Embedding-based similarity was applied to select relevant context chunks when multiple documents were present. This was achieved using a **RAG (Retrieval-Augmented Generation)** approach for embedding search, where relevant context was retrieved before generating responses.

### 3. Model Evaluation
The model was evaluated using **BERTScore** to measure the similarity between predicted and true answers. The following metrics were considered:
- **F1-score** (for balancing precision and recall)

**Average BERTScore F1:** 0.901 (high agreement between predicted and actual answers).

### 4. Example Interactions

- **Question:** What is blood pressure?
  **Answer:** The ranges in the table are blood pressure guides for adults who do not have any short-term serious illnesses.
People with diabetes or chronic kidney disease should keep their blood pressure below 130/80 mmHg.
Blood pressure is the force of blood pushing against the walls of the blood vessels as the heart pumps blood.
If your blood pressure rises and stays high over time, its called high blood pressure.
High blood pressure is dangerous because it makes the heart work too hard

- **Question:** What is diabetes?
  **Answer:** Diabetes means your blood glucose (often called blood sugar) is too high.
Your blood always has some glucose in it because your body needs glucose for energy to keep you going.
But too much glucose in the blood isn't good for your health.
Glucose comes from the food you eat and is also made in your liver and muscles.
Your blood carries the glucose to all of the cells in your body. Insulin is a chemical (a hormone) made by the pancreas.
The pancreas releases insulin into the blood.

- **Question:** What is glaucoma?
  **Answer:** Glaucoma is a group of diseases that can damage the eye's optic nerve and result in vision loss and blindness.
The most common form of the disease is open-angle glaucoma. With early treatment, you can often protect your eyes
against serious vision loss. (Watch the video to learn more about glaucoma. To enlarge the video, click the brackets
in the lower right-hand corner. To reduce the video, press the Escape (Esc) button on your keyboard.)

## Model Performance

- **Strengths:** The combination of BERT and GPT-2 provides both accurate and fluent responses. The use of embeddings helps select the most relevant information, improving the precision of answers. The fine-tuning approach, with 80% of the data for training and 20% for validation, ensures good model generalization and mitigates overfitting.
- **Weaknesses:** The model can struggle with very complex medical queries requiring deeper medical knowledge or extensive reasoning beyond the dataset context. Test different fine tunning approche is too long for 2 day task, maybe ski

## Potential Improvements

- **More Powerful Models:** An improvement could be to experiment with more powerful models like **GPT-4**. This would likely increase the overall accuracy and capability of handling more complex or nuanced medical questions.
- **Vector Database for Embedding Search:** The performance of the retrieval stage could be enhanced by using a **vector database like Milvus** to efficiently manage and search through large-scale embeddings. This would significantly speed up and improve the relevance of context retrieval during the **RAG (Retrieval-Augmented Generation)** phase.
- **Context length handling:** The current implementation truncates inputs to fit into model constraints. Future iterations could use sliding windows or long-form transformers like Longformer for better handling of longer contexts.
- **Data Augmentation:** Incorporating additional medical datasets (e.g., MIMIC-III, PubMed) could improve the model’s accuracy by diversifying the training data.
- **Fine-tuning GPT-2:** Fine-tuning GPT-2 on a edical dialogue dataset could generate more context-aware, conversational answers.
- **Data Cleaning**: I observed a lot of irrelevant content, including HTML code and non-medical text, which required cleaning. We can see it on glaucoma answer.
- **Testing with Ollama and other LLM models**


**In a real-world scenario, I would approach this problem using RAG (Retrieval-Augmented Generation) with more powerful embeddings, such as OpenAI Embeddings, with their vectors stored in a vector database like Milvus. I would then use a more robust model like GPT-4o. The entire workflow would be implemented using LangChain for efficient orchestration, and I would utilize its RetrievalQA to improve the accuracy and performance of the answers provided. To monitor experiments, fine-tuning, prompts engineering, and parameters search, I would leverage tools like WandB**

**Tuning the params and testing different fine-tuning approaches is not feasible in this case, as it would take a considerable amount of time, which would impact the two-day timeframe I have for this task. However, in a real-world scenario, I would have access to more computational resources and time, allowing me to perform such experiments and fine-tuning tests more thoroughly.**

**The code can be better structured with improved software engineering, using Object-Oriented Programming (OOP), unit tests, separation of responsibilities, and modular organization.**