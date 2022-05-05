# Conversational Bot, that will answer your Data Science interview questions :)
Dialog System, that performs One-Class Classification, Question Answering, and Natural Language Generation tasks

 ![](https://github.com/Teasotea/DialogSystem/blob/main/img/greeting_cl_example.png)

## Task
The task was to create a bot, which would welcome the user only if he or she sends the greetings first. The bot should tell the right answers for Data Science interview Questions if they exist in the Database. Otherwise, the bot should generate answers with the help of NLG techniques. If the user wants to stop the conversation, the bot should classify that intention and say 'See you soon! Bye!'


## Project Details
 Development of the project consisted of 4 main parts:
  1) One-Class Classification: Preprocessing text and building 2 separate models for greeting and quitting conservation intend classification. Gathering datasets for that purpose. Solution based on `OneClassSVM` model from sklearn. Evaluation of model performance with f1 score metric
  2) Question Answering: retrieval of question intend with text summarization model `google/pegasus-xsum`, applying Sentence Transformers `all-mpnet-base-v2` and cosine similarity for comparing questions with the given context, information retrieval with BERT model, pre-trained on SQUAD2 Dataset `deepset/bert-base-cased-squad2` for answering the question
  3) Natural Language Generation: using `microsoft/DialoGPT-medium` to generate text
  4) Development of an end-user Dialog System, that can perform the conversion using models decribed above


![](https://github.com/Teasotea/DialogSystem/blob/main/img/chatbot_results.png)
