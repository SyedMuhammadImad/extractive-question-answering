"""Extractive QA inference and SQuAD-compatible exact match/token F1."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string

DEFAULT_MODEL = 'distilbert/distilbert-base-cased-distilled-squad'

def normalize(text):
    text = str(text).lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    return ' '.join(text.split())

def exact_match(prediction, reference):
    return float(normalize(prediction) == normalize(reference))

def token_f1(prediction, reference):
    pred, ref = normalize(prediction).split(), normalize(reference).split()
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)

def load_model(model_name=DEFAULT_MODEL):
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    import torch
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=False, trust_remote_code=False)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, token=False, trust_remote_code=False)
    model.eval()

    def predict(*, context, question):
        encoded = tokenizer(question, context, max_length=384, stride=128,
                            truncation='only_second', return_overflowing_tokens=True,
                            return_offsets_mapping=True, padding=True, return_tensors='pt')
        offsets = encoded.pop('offset_mapping')
        encoded.pop('overflow_to_sample_mapping')
        with torch.inference_mode():
            output = model(**encoded)
        best = {'answer': '', 'score': 0.0, 'start': 0, 'end': 0}
        for window in range(len(offsets)):
            context_mask = torch.tensor([s == 1 for s in encoded.sequence_ids(window)])
            start = output.start_logits[window].masked_fill(~context_mask, float('-inf')).softmax(-1)
            end = output.end_logits[window].masked_fill(~context_mask, float('-inf')).softmax(-1)
            for left in start.topk(min(20, len(start))).indices.tolist():
                for right in end.topk(min(20, len(end))).indices.tolist():
                    if right < left or right - left >= 30 or not context_mask[left] or not context_mask[right]:
                        continue
                    a, b = int(offsets[window, left, 0]), int(offsets[window, right, 1])
                    score = float(start[left] * end[right])
                    if b > a and score > best['score']:
                        best = {'answer': context[a:b], 'score': score, 'start': a, 'end': b}
        return best
    return predict

def answer(model, context, question):
    if not isinstance(context, str) or not isinstance(question, str) or not context.strip() or not question.strip():
        raise ValueError('Both context and question must be non-empty text.')
    if len(context) > 1_000_000 or len(question) > 2000:
        raise ValueError('Input exceeds the supported size.')
    return model(context=context, question=question)

def squad_rows(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    for article in data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                yield {'id':qa['id'], 'context':paragraph['context'], 'question':qa['question'],
                       'answers':[a['text'] for a in qa['answers']] or ['']}

def evaluate(model, rows):
    predictions=[]
    for row in rows:
        result=answer(model, row['context'], row['question'])
        refs=row['answers']
        predictions.append({'id':row['id'], 'prediction':result['answer'],
                            'exact_match':max(exact_match(result['answer'],r) for r in refs),
                            'f1':max(token_f1(result['answer'],r) for r in refs)})
    if not predictions:raise ValueError('Evaluation requires at least one example.')
    return {'count':len(predictions), 'exact_match':sum(p['exact_match'] for p in predictions)/len(predictions),
            'f1':sum(p['f1'] for p in predictions)/len(predictions), 'predictions':predictions}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,required=True,help='Official SQuAD dev-v1.1.json')
    parser.add_argument('--models',nargs='+',default=[DEFAULT_MODEL])
    parser.add_argument('--limit',type=int,default=100)
    parser.add_argument('--output',type=Path,default=Path('qa_metrics.json'))
    args=parser.parse_args()
    if args.limit<=0:parser.error('--limit must be positive')
    from random import Random
    rows=list(squad_rows(args.data));Random(42).shuffle(rows);rows=rows[:args.limit]
    results={}
    for name in args.models:
        model=load_model(name)
        results[name]=evaluate(model,rows)
        del model
    args.output.write_text(json.dumps({'dataset':'SQuAD v1.1 development split','seed':42,
                                      'evaluation':results},indent=2),encoding='utf-8')
    print(json.dumps({n:{k:v for k,v in r.items() if k!='predictions'} for n,r in results.items()}))

if __name__=='__main__':main()
