#DataLoad
import pandas as pd

bond_path='/content/bond_report_master.csv'
news_path='/content/news_posclean_dirty.csv'
fomc_path='/content/FOMC_full_sentence_level_with_sections_3.csv'

df_bond=pd.read_csv(bond_path, usecols=['doc_id','cleaned_text'])
df_news=pd.read_csv(news_path, usecols=['doc_id','cleaned_text'])
df_fomc=pd.read_csv(fomc_path, usecols=['doc_id','cleaned_text'])

df=pd.concat([df_bond, df_news, df_fomc],axis=0)

df_5_gram=pd.read_csv('/content/fivegram_collocations.csv')
df_2_gram=pd.read_csv('/content/bigram_collocations.csv')
df_3_gram=pd.read_csv('/content/trigram_collocations.csv')
df_4_gram=pd.read_csv('/content/fourgram_collocations.csv')
df_n_gram=pd.concat([df_2_gram, df_3_gram, df_4_gram, df_5_gram])

#freq 15 이상만 남기기(논문 기준)
df_n_gram_freq15=df_n_gram[df_n_gram['freq']>=15]

#문서 재구성 (doc_id 기준 'cleaned_text' 연결)
df['cleaned_text'] = df['cleaned_text'].fillna('')
df_sum = df.groupby('doc_id')['cleaned_text'].agg(lambda x: ' '.join(x)).reset_index()
df_sum.rename(columns={'cleaned_text': 'full_document'}, inplace=True)
df_sum['full_document'] = df_sum['full_document'].apply(lambda x: ', '.join(set(x.split())))

#distilling by collocation
collocation_text = df_n_gram_freq15['ngram'].apply(lambda x: ', '.join(set(x.split())))
collocation_text=collocation_text.to_list()

#연어 모음 불러오기
import pickle

file_path='/content/collocation_list.pkl'
# 'rb': 읽기 모드 (Read Binary)
with open(file_path, 'rb') as f:
    collocation_text_list = pickle.load(f)
collocation_set = set(collocation_text_list)

def collocation_words(col):
    words = str(col).split(', ')
    return ' '.join([word for word in words if word in collocation_set])

df_sum['distilled_by_collocation']=df_sum['lemmatized_text'].apply(collocation_words)

df_sum['distilled_by_collocation']=df_sum['lemmatized_text'].apply(collocation_words)


#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# 2. TfidfVectorizer 적용
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_sum['distilled_by_collocation'])
# tfidf_matrix = vectorizer.fit_transform(df_sum['cleaned_text'])

feature_names = vectorizer.get_feature_names_out()

# 3. TF-IDF over Threshold per Doc
import numpy as np

def extract_filtered_keywords(doc_index, tfidf_matrix, feature_names, min_score=0.5):
    # doc_index가 행렬 범위를 벗어나지 않는지 방어 코드 (선택 사항)
    if doc_index >= tfidf_matrix.shape[0]:
        return ""

    # 해당 문서의 점수 추출
    vector = tfidf_matrix.getrow(doc_index).toarray().flatten()

    # 1. 점수 필터링 (min_score 이상인 것만)
    filtered_indices = np.where(vector >= min_score)[0]

    # 해당되는 단어가 없으면 빈 문자열 반환
    if len(filtered_indices) == 0:
        return ""

    # 2. 점수 내림차순 정렬
    # vector[filtered_indices]로 값만 먼저 뽑고, 그 안에서 argsort를 수행
    sorted_indices_local = np.argsort(vector[filtered_indices])[::-1]
    sorted_indices = filtered_indices[sorted_indices_local]

    # 3. 문자열 생성
    result_list = [
        f"{feature_names[i]}: {vector[i]:.3f}"
        for i in sorted_indices
    ]

    return ', '.join(result_list)

# --- 함수 적용 ---
df_sum = df_sum.reset_index(drop=True)

df_sum['tf_idf_over0.5'] = [
    extract_filtered_keywords(i, tfidf_matrix, feature_names, min_score=0.5)
    for i in range(len(df_sum))
]

df_mapping = df_sum[['doc_id', 'tf_idf_over0.5', 'distilled_by_collocation']]

# print(df_mapping.shape)
# print(df_mapping.head())

# 3. TF-IDF TOP N Per Doc
def extract_top_keywords(doc_index, top_n=50):
    vector = tfidf_matrix.getrow(doc_index).toarray().flatten()
    top_indices = vector.argsort()[-top_n:][::-1]

    # 상위 N개 단어만 리스트로 반환
    top_words = [f"{feature_names[i]}: {vector[i]:.3f}" for i in top_indices]
    return ', '.join(top_words)

# 4. df_sum에 'top_keywords' 컬럼 추가
df_mapping['tf_idf_top50'] = [extract_top_keywords(i) for i in range(len(df_mapping))]

# # 5. doc_id를 기준으로 원본 df에 병합 (매핑)
# # df_sum의 'doc_id'와 'top_keywords'만 필요
df_mapping = df_mapping[['doc_id', 'tf_idf_over0.5', 'tf_idf_top50', 'distilled_by_collocation']]

# 'doc_id'가 일치하는 모든 문장(df의 행)에 해당 문서의 top_keywords를 할당
df = pd.merge(df, df_mapping, on='doc_id', how='left')

df.to_csv('df_master_TM_proj.csv', index=False)
