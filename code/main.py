import string
import torch
import numpy as np
from transformers import AutoTokenizer, BertForQuestionAnswering, GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, \
    Trainer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score


def preprocess_data(examples):
    inputs = tokenizer(examples['question'], examples['context'], truncation=True, padding='max_length',
                       max_length=512, return_tensors="pt", return_attention_mask=True)
    start_positions = []
    end_positions = []
    for i in range(len(examples['context'])):
        start_idx = 0
        end_idx = len(examples['context'][i]) - 1
        start_positions.append(start_idx)
        end_positions.append(end_idx)
    inputs['start_positions'] = start_positions
    inputs['end_positions'] = end_positions
    return inputs


def get_embedding_answer(question):
    new_question_embedding = embedding_model.encode([question])
    cosine_similarities = cosine_similarity(new_question_embedding, questions_embeddings)
    top_indices = np.argsort(cosine_similarities[0])[::-1][:5]
    top_answers = [dataset['train']['context'][i] for i in top_indices]
    context = " ".join(top_answers)
    return [context[i:i + 512] for i in range(0, len(context), 512)]


def get_qa_bert_answer(question, context):
    inputs = tokenizer(question, context, truncation=True, padding='max_length',
                       max_length=512, return_tensors="pt", return_attention_mask=True)

    with torch.no_grad():
        outputs = qa_model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    answer_tokens = inputs.input_ids[0][start_index:end_index + 1]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True)


def get_gpt_answer(question, context):
    input_text = (
        f"You are a doctor and an expert in medical question answering. "
        f"Read the question and relevant information below. "
        f"Provide a clear, concise, and medically accurate answer."
        f"Avoid unrelated information\n\n"
        f"Question: {question}\n"
        f"Relevant information: {context[:1024]}\n\n"
        f"Answer:"
    )
    inputs = gpt_tokenizer(
        input_text,
        return_tensors='pt',
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_attention_mask=True
    )
    outputs = gpt_model.generate(input_ids=inputs["input_ids"],
                                 attention_mask=inputs["attention_mask"],
                                 max_new_tokens=256,
                                 num_return_sequences=1,
                                 pad_token_id=gpt_tokenizer.pad_token_id,
                                 temperature=0.7,
                                 top_p=0.9,
                                 do_sample=True,
                                 repetition_penalty=1.2)
    response = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(input_text, "").strip()
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    return response


def ask_question(question):
    context_chunks = get_embedding_answer(question)
    context = " ".join(context_chunks)

    bert_answer = get_qa_bert_answer(question, context).strip(string.punctuation + ' ')

    return get_gpt_answer(question, bert_answer).strip(string.punctuation + ' ')


def compute_bertscore(predicted_answer, true_answer):
    P, R, F1 = score([predicted_answer], [true_answer], lang='en')
    return F1.mean().item()


qa_model_name, embedding_model_name = 'dmis-lab/biobert-v1.1', 'all-MiniLM-L6-v2'

dataset = load_dataset('csv', data_files="../data/mle_screening_dataset.csv")  # Use o seu dataset específico de QA
tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

dataset['train'] = dataset['train'].filter(
    lambda x: x['question'] is not None and x['context'] is not None and x['question'].strip() != "" and x[
        'context'].strip() != "")
dataset_split = dataset['train'].train_test_split(test_size=0.2)
train_dataset = dataset_split['train'].map(preprocess_data, batched=True)
eval_dataset = dataset_split['test'].map(preprocess_data, batched=True)

embedding_model = SentenceTransformer(embedding_model_name)
qa_model = BertForQuestionAnswering.from_pretrained(qa_model_name)
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

questions_embeddings = embedding_model.encode(dataset['train']['question'])

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=qa_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

print(ask_question('What is blood pressure?'))
print(ask_question('What is diabetes?'))
print(ask_question('What is Glaucoma?'))

pred_answers = []
for q, a in zip(eval_dataset['question'], eval_dataset['context']):
    pred_answers.append(ask_question(q))
bertscore_f1_scores = [compute_bertscore(pred, true) for pred, true in zip(pred_answers, eval_dataset['context'])]
average_bertscore_f1 = np.mean(bertscore_f1_scores)
print(print(len(pred_answers)), average_bertscore_f1)   # 0.9009693738928076
