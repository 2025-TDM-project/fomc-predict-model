import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. 설정 및 경로
PATHS = {
    'bond': '/content/bond_report_master.csv',
    'news': '/content/news_posclean_dirty.csv',
    'fomc': '/content/FOMC_full_sentence_level_with_sections_3.csv',
    'collocation_pickle': '/content/collocation_list.pkl',
    'output': 'df_master_TM_proj.csv'
}

# 2. 데이터 로드 및 병합 (사용할 컬럼 추가)
def load_and_merge_documents(target_col):
    cols = ['doc_id', target_col]
    
    # 각 파일 로드 시 target_col이 있는지 확인 필요
    df_bond = pd.read_csv(PATHS['bond'], usecols=cols)
    df_news = pd.read_csv(PATHS['news'], usecols=cols)
    df_fomc = pd.read_csv(PATHS['fomc'], usecols=cols)

    df_merged = pd.concat([df_bond, df_news, df_fomc], axis=0, ignore_index=True)
    df_merged[target_col] = df_merged[target_col].fillna('')
    
    return df_merged

def load_collocation_set():
    with open(PATHS['collocation_pickle'], 'rb') as f:
        collocation_text_list = pickle.load(f)
    return set(collocation_text_list)

# 3. 텍스트 전처리 및 그룹화
def preprocess_documents(df, target_col):
    """doc_id 기준으로 텍스트를 합치고 중복 단어를 제거합니다."""
    df_sum = df.groupby('doc_id')[target_col].agg(lambda x: ' '.join(x)).reset_index()
    
    # TF-IDF 분석을 위해 고유 단어들로 재구성
    df_sum['full_document'] = df_sum[target_col].apply(lambda x: ', '.join(set(x.split())))
    return df_sum

def filter_by_collocation(text, collocation_set):
    """연어(Collocation) 리스트에 포함된 단어만 추출합니다."""
    words = str(text).split(', ')
    filtered_words = [word for word in words if word in collocation_set]
    return ' '.join(filtered_words)

# 4. TF-IDF 분석 함수
def get_keywords_over_threshold(doc_idx, tfidf_matrix, feature_names, min_score=0.5):
    vector = tfidf_matrix[doc_idx].toarray().flatten()
    indices = np.where(vector >= min_score)[0]
    if len(indices) == 0: return ""
    # 점수 내림차순 정렬
    sorted_indices = indices[np.argsort(vector[indices])[::-1]]
    return ', '.join([f"{feature_names[i]}: {vector[i]:.3f}" for i in sorted_indices])

def get_top_n_keywords(doc_idx, tfidf_matrix, feature_names, top_n=50):
    vector = tfidf_matrix[doc_idx].toarray().flatten()
    # 상위 N개 추출
    top_indices = vector.argsort()[-top_n:][::-1]
    # 0점인 단어 제외
    top_indices = [i for i in top_indices if vector[i] > 0]
    return ', '.join([f"{feature_names[i]}: {vector[i]:.3f}" for i in top_indices])

# 5. 메인 실행 흐름
def main(target_col='cleaned_text_lemma'):
    print(f"1. Loading Documents using column: {target_col}...")
    df_origin = load_and_merge_documents(target_col)
    
    print("2. Grouping Documents by ID...")
    df_sum = preprocess_documents(df_origin, target_col)
    
    print("3. Filtering by Collocations...")
    collocation_set = load_collocation_set()
    df_sum['distilled_by_collocation'] = df_sum['full_document'].apply(
        lambda x: filter_by_collocation(x, collocation_set)
    )

    print("4. Calculating TF-IDF...")
    vectorizer = TfidfVectorizer()
    # 연어 필터링이 완료된 텍스트로 벡터화 수행
    tfidf_matrix = vectorizer.fit_transform(df_sum['distilled_by_collocation'])
    feature_names = vectorizer.get_feature_names_out()

    print("5. Extracting Top 50 & Threshold 0.5 Keywords...")
    # 문서별 결과 생성
    df_sum['tf_idf_over0.5'] = [
        get_keywords_over_threshold(i, tfidf_matrix, feature_names, min_score=0.5) 
        for i in range(len(df_sum))
    ]
    df_sum['tf_idf_top50'] = [
        get_top_n_keywords(i, tfidf_matrix, feature_names, top_n=50) 
        for i in range(len(df_sum))
    ]

    print("6. Merging and Saving...")
    # 필요한 결과 컬럼만 추출하여 원본과 매핑
    df_mapping = df_sum[['doc_id', 'tf_idf_over0.5', 'tf_idf_top50']]
    df_final = pd.merge(df_origin, df_mapping, on='doc_id', how='left')
    
    df_final.to_csv(PATHS['output'], index=False)
    print(f"Success! File saved as: {PATHS['output']}")

if __name__ == "__main__":
    # 'cleaned_text_lemma' 컬럼을 기준으로 실행
    main(target_col='cleaned_text_lemma')
