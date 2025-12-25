prompt_dict = {
    'qa': {
        'naive_LLM_system':
        'You are a helpful assistant!\nAnswer the following question based on your internal knowledge.\nOnly give me the answer and do not output any other words.',
        'naive_LLM_user':
        'Question: {question}\nAnswer:',
        'CoN_notes_system':
        'Task: Read the question and passages. Write concise reading notes summarizing key facts relevant to answering the question. Do not provide the final answer. Use short bullet points.',
        'CoN_notes_user':
        'Passages:\n{passages}\n\nQuestion: {question}\nNotes:',
        'CoN_answer_system':
        'Only output the final answer as one or few words. Do not output any other words. Use the provided reading notes and the passages to determine the answer. If no passage is relevant, answer based on internal knowledge.',
        'CoN_answer_user':
        'Passages:\n{passages}\n\nNotes:\n{notes}\n\nQuestion: {question}\nAnswer:',
        'CoN_answer_system_only_notes':
        'Only output the final answer as one or few words. Do not output any other words. Use the provided reading notes to determine the answer. If no note is relevant, answer based on internal knowledge.',
        'CoN_answer_user_only_notes':
        'Notes:\n{notes}\n\nQuestion: {question}\nAnswer:',
        'RAG_system':
        'You are a helpful assistant!',
        'RAG_user': 
        'Given the following information: \n{paras}\n\nAnswer the following question based on the given information or your internal knowledge with one or few words without the source.\nQuestion: {question}\nAnswer: {answer}',
    },
    'vicuna': {
        'naive_LLM_system':
        'Answer the question using your internal knowledge.\nOnly output the final answer. Do not include explanations.',
        'naive_LLM_user':
        'Question: {question}',
    }
}
