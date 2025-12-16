prompt_dict = {
    'qa': {
        'naive_LLM':
        'Answer the following question based on your internal knowledge with one or few words.\n\nQuestion: {question}\nAnswer:',
        'naive_RAG':
        'Answer the question based on the given document.\nOnly give me the answer and do not output any other words.\n\nThe following are given documents. \n{paras}.\n\nQuestion: {question}\nAnswer: {answer}',
        'naive_RAG_system':
        'Answer the question based on the given document.\nOnly give me the answer and do not output any other words.\n\nThe following are given documents. \n{paras}',
        'naive_RAG_user':
        'Question: {question}\nAnswer: {answer}',
        'CoN_system':
        'You are a helpful assistant!',
        'CoN_user':
        'Task Description: \n 1. Read the given question and five Wikipedia passages to gather relevant information.\n2. Write reading notes summarizing the key points from these passages.\n3. Discuss the relevance of the given question and Wikipedia passages.\n4. If some passages are relevant to the given question, provide a brief answer based on the passages. \n5. If no passage is relevant, direcly provide answer without considering the passages.\n6. Only give me the answer and do not output any other words.\n\nPassages:\n{passages}\n\nQuestion: {question}',
        'CoN_notes_system':
        'Task: Read the question and passages. Write concise reading notes summarizing key facts relevant to answering the question. Do not provide the final answer. Use short bullet points.\n\nPassages:\n{passages}',
        'CoN_notes_user':
        'Question: {question}\nNotes:',
        'CoN_answer_system':
        'Only output the final answer as one or few words. Do not output any other words. Use the provided reading notes and the passages to determine the answer. If no passage is relevant, answer based on internal knowledge.',
        'CoN_answer_user':
        'Passages:\n{passages}\nNotes:\n{notes}\nQuestion: {question}\nAnswer:',
        'RAG_system':
        'You are a helpful assistant!',
        'RAG_user': 
        'Given the following information: \n{paras}\n\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source.\nQuestion: {question}\nAnswer: {answer}',
    },
}
