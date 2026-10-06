"""Local Streamlit UI; model loading happens only when an answer is requested."""
import pandas as pd
import streamlit as st
from qa import DEFAULT_MODEL, load_model, answer

st.set_page_config(page_title='Question Answering System',layout='wide')
st.title('Question Answering System')
st.caption('Extracts an answer from your context. The score is a model score, not verified correctness.')
model_name=DEFAULT_MODEL
st.sidebar.caption('DistilBERT trained for answerable SQuAD questions. Supply context containing the answer.')
@st.cache_resource
def cached_model(name):return load_model(name)
context=st.text_area('Context')
question=st.text_input('Question')
if st.button('Get Answer'):
    if not context.strip() or not question.strip():st.warning('Enter both context and question.')
    else:
        try:
            result=answer(cached_model(model_name),context,question)
            st.write(result['answer']);st.metric('Model score',f"{result['score']:.3f}")
        except Exception as exc:st.error(str(exc))
st.subheader('Batch CSV')
uploaded=st.file_uploader('CSV with context and question columns',type=['csv'])
if uploaded is not None:
    try:
        data=pd.read_csv(uploaded)
        if not {'context','question'}<=set(data.columns):raise ValueError('Required columns: context, question.')
        if len(data)>100:raise ValueError('Batch limit is 100 rows.')
        if st.button('Process Batch'):
            model=cached_model(model_name)
            rows=[]
            for _,row in data.iterrows():
                try:result=answer(model,row['context'],row['question']);rows.append(result)
                except ValueError as exc:rows.append({'answer':'','error':str(exc)})
            output=pd.DataFrame(rows)
            st.dataframe(output)
            st.download_button('Download answers',output.to_csv(index=False),'qa_results.csv','text/csv')
    except Exception as exc:st.error(str(exc))
