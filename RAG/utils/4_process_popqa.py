import json
import os
import argparse

def process_popqa(input_file, output_file):
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for idx, line in enumerate(f_in):
            if not line.strip():
                continue
                
            data = json.loads(line)
            
            # 1. Create id starting from 0
            new_entry = {
                'id': idx,
                'question': data.get('question', ''),
                'answers': data.get('answers', []),
                'positive_passages': [],
                'negative_passages': []
            }
            
            # 2. Process positive_ctxs
            if 'positive_ctxs' in data:
                for ctx in data['positive_ctxs']:
                    # Ensure strictly True (not just truthy, though usually boolean in JSON)
                    if ctx.get('has_answer') is True:
                        new_entry['positive_passages'].append(ctx)
            
            # 3. Process ctxs_dpr
            if 'ctxs_dpr' in data:
                for ctx in data['ctxs_dpr']:
                    # Ensure strictly False
                    if ctx.get('has_answer') is False:
                        new_entry['negative_passages'].append(ctx)
            
            f_out.write(json.dumps(new_entry) + '\n')
            count += 1
            
    print(f"Processed {count} lines.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PopQA dataset")
    parser.add_argument("--input_file", type=str, default="/root/shared-nvme/RAG-llm/RAG/data/PopQA/PopQA.jsonl", help="Path to input file")
    parser.add_argument("--output_file", type=str, default="/root/shared-nvme/RAG-llm/RAG/data/PopQA/PopQA_processed.jsonl", help="Path to output file")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    process_popqa(args.input_file, args.output_file)
