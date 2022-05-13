# Conversational Bot, that will answer your Data Science interview questions :)
Dialog System, that performs One-Class Classification, Question Answering, Sentence Similarity, and Natural Language Generation tasks

![](https://github.com/Teasotea/DialogSystem/blob/main/img/chatbot_results.png)

## Task
The task was to create a bot, which would welcome the user only if he or she sends the greetings first. The bot should tell the right answers for Data Science interview questions if they exist in the Database. Otherwise, the bot should generate answers with the help of NLG techniques. If the user wants to stop the conversation, the bot should classify that intention and say 'See you soon! Bye!'


## Project Details
 Development of the project consisted of 4 main parts:
  1) One-Class Classification: Preprocessing text and building 2 separate models for greeting and quitting conservation intend classification. Gathering datasets for that purpose. Solution based on `OneClassSVM` model from sklearn. Evaluation of model performance with f1 score metric
  2) Question Answering: retrieval of question intend with text summarization model `google/pegasus-xsum`, applying Sentence Transformers `all-mpnet-base-v2` and cosine similarity for comparing questions with the given context, information retrieval with BERT model, pre-trained on SQUAD2 Dataset `deepset/bert-base-cased-squad2` for answering the question
  3) Natural Language Generation: using `microsoft/DialoGPT-medium` to generate text
  4) Development of an end-user Dialog System, that can perform the conversion using models decribed above

## Datasets
1) For the task of One-Class Classification, 2 datasets were created: '[Greetings](https://github.com/Teasotea/DialogSystem/blob/main/data/greet.csv)' and '[Goodbyes](https://raw.githubusercontent.com/Teasotea/DialogSystem/main/data/goodbyes.csv)'. They consist of common expressions of greetings and farewells. To improve the performance of the model, the datasets could be extended.
2) For the QA part in is used the [dataset](https://raw.githubusercontent.com/Kizuna-Cheng/Data_Science_Interviews_NLP/main/data.csv) of Data Science interview questions. It has only 323 rows. For future improvements to the information retrieval part of the project, it is worthy to find a bigger one.

P.S. To make transformer models more specific to our use case, it is worthy to fine-tune them on datasets related to computer science topics. It can be also a scrapped Quora/Stackoverflow questions and answers.



## Implementation
 ![](https://github.com/Teasotea/DialogSystem/blob/main/img/chatbot_diagram.png)
 
How does class ChatBot work?

1) Wait for the user's input to start the conversation
2) Classify whether the message has the intention to end the conversation with the `OneClassSVM` model: if yes - the chat ends. 
3) Classify whether the message is a greeting with the `OneClassSVM` model: if yes - randomly choose 1 of 4 sentences and say 'hello'
4) If the message is not a greeting - extract the main idea of the sentence with `pegasus-xsum` summarization model. It helps to reduce redundant words, which make text search for similar sentences more difficult.
5) Check whether the Data Science interview questions database has a similar question (by building word embeddings with the `Sentence Transformers` model and comparing questions with `cosine similarity`). 
6) If the user's input is a question from the base - call the answer_with_BERT() function and perform the information retrieval (with Bert, pre-trained on SQUAD dataset `bert-base-cased-squad2`)
7) If a similar question to the input is not found in the base - tokenize it, save it in history_ids, and launch the `DialoGPT` model to generate the answer
8) Repeat steps until the intention to quit found

## Notebooks
* [`ConversationalAI.ipynb`](https://github.com/Teasotea/DialogSystem/blob/main/ConversationalAI.ipynb) ([nbviewer](https://github.com/Teasotea/DialogSystem/blob/main/ConversationalAI.ipynb)) - Notebook version, in which Text Summarization Model wasn't used: it works a little bit faster, but less accurate in searching for answers.
* [`ConversationalAI_v2.ipynb`](https://github.com/Teasotea/DialogSystem/blob/main/ConversationalAI_v2.ipynb) ([nbviewer](https://github.com/Teasotea/DialogSystem/blob/main/ConversationalAI_v2.ipynb)) -  Notebook version, with `pegasus-xsum` Text Summarization Model. The bot works a little bit slower, but performs better with Text Similarity task.

Text Summarisation Model extracts the core idea of message, which solves the problem of comparing different sentences with similar meaning:
`What does linear regression stand for?`
`What is linear regression`
`Tell me pls, what is linear regression?`

