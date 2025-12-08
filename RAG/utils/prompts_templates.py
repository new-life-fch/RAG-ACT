prompt_dict = {
    'qa': {
        'naive_LLM':
        'Answer the following question based on your internal knowledge with one or few words.\n\nQuestion: {question}\nAnswer: ',
        'naive_RAG':
        'Answer the question based on the given document.\nOnly give me the answer and do not output any other words."\n\nThe following are given documents. \n{paras}.\n\nQuestion: {question}\nAnswer: {answer}',
        'naive_RAG_system':
        'Answer the question based on the given document.\nOnly give me the answer and do not output any other words."\n\nThe following are given documents. \n{paras}',
        'naive_RAG_user':
        'Question: {question}\nAnswer: {answer}',
        'CoN_system':
        'Task Description: \n 1. Read the given question and five Wikipedia passages to gather relevant information.\n2. Write reading notes summarizing the key points from these passages.\n3. Discuss the relevance of the given question and Wikipedia passages.\n4. If some passages are relevant to the given question, provide a brief answer based on the passages. \n5. If no passage is relevant, direcly provide answer without considering the passages.\n\nPassages:\n{passages}',
        'CoN_user':
        '{question}',
    },
}