import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. 설정 및 경로 (동일)
PATHS = {
    'bond': '/content/bond_report_master.csv',
    'news': '/content/news_posclean_dirty.csv',
    'fomc': '/content/FOMC_full_sentence_level_with_sections_3.csv',
    'collocation_pickle': '/content/collocation_list.pkl',
    'output': 'df_master_TM_proj.csv'
}

# 2. 데이터 로드 및 병합 (동일)
def load_and_merge_documents():
    cols = ['doc_id', 'cleaned_text']
    df_bond = pd.read_csv(PATHS['bond'], usecols=cols)
    df_news = pd.read_csv(PATHS['news'], usecols=cols)
    df_fomc = pd.read_csv(PATHS['fomc'], usecols=cols)
    df_merged = pd.concat([df_bond, df_news, df_fomc], axis=0, ignore_index=True)
    df_merged['cleaned_text'] = df_merged['cleaned_text'].fillna('')
    return df_merged

def load_collocation_set():
    with open(PATHS['collocation_pickle'], 'rb') as f:
        collocation_text_list = pickle.load(f)
    return set(collocation_text_list)

# ==========================================
# 3. 수정된 전처리 함수 (target_column 매개변수 추가)
# ==========================================
def preprocess_documents(df, target_column):
    """지정된 컬럼(target_column)을 doc_id 기준으로 그룹화합니다."""
    # 지정된 컬럼에 대해 join 수행
    df_sum = df.groupby('doc_id')[target_column].agg(lambda x: ' '.join(x)).reset_index()
    
    # 결과 컬럼명을 'full_document'로 통일
    df_sum.rename(columns={target_column: 'full_document'}, inplace=True)
    
    # 중복 단어 제거
    df_sum['full_document'] = df_sum['full_document'].apply(lambda x: ', '.join(set(x.split())))
    return df_sum

# 4. 분석 함수 (동일)
def get_keywords_over_threshold(doc_idx, tfidf_matrix, feature_names, min_score=0.5):
    vector = tfidf_matrix[doc_idx].toarray().flatten()
    filtered_indices = np.where(vector >= min_score)[0]
    if len(filtered_indices) == 0: return ""
    sorted_indices = filtered_indices[np.argsort(vector[filtered_indices])[::-1]]
    return ', '.join([f"{feature_names[i]}: {vector[i]:.3f}" for i in sorted_indices])

def get_top_n_keywords(doc_idx, tfidf_matrix, feature_names, top_n=50):
    vector = tfidf_matrix[doc_idx].toarray().flatten()
    top_indices = vector.argsort()[-top_n:][::-1]
    return ', '.join([f"{feature_names[i]}: {vector[i]:.3f}" for i in top_indices])

def filter_by_collocation(text, collocation_set):
    words = str(text).split(', ')
    return ' '.join([word for word in words if word in collocation_set])

# ==========================================
# 5. 메인 실행 흐름 (매개변수 추가)
# ==========================================
def main(target_col='cleaned_text'): # 분석하고 싶은 컬럼명을 인자로 받음
    print(f"--- TF-IDF Analysis for Column: {target_col} ---")
    
    print("1. Loading Documents...")
    df_origin = load_and_merge_documents()
    
    print(f"2. Grouping Documents by ID using '{target_col}'...")
    # 전처리 시 타겟 컬럼을 넘겨줌
    df_sum = preprocess_documents(df_origin, target_column=target_col)
    
    print("3. Loading Collocations & Filtering...")
    collocation_set = load_collocation_set()
    df_sum['distilled_by_collocation'] = df_sum['full_document'].apply(
        lambda x: filter_by_collocation(x, collocation_set)
    )

    print("4. Running TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer()
    # 필터링된 텍스트로 TF-IDF 수행
    tfidf_matrix = vectorizer.fit_transform(df_sum['distilled_by_collocation'])
    feature_names = vectorizer.get_feature_names_out()

    print("5. Extracting Keywords...")
    df_sum['tf_idf_over0.5'] = [
        get_keywords_over_threshold(i, tfidf_matrix, feature_names, min_score=0.5) 
        for i in range(len(df_sum))
    ]
    df_sum['tf_idf_top50'] = [
        get_top_n_keywords(i, tfidf_matrix, feature_names, top_n=50) 
        for i in range(len(df_sum))
    ]

    print("6. Merging Results & Saving...")
    df_mapping = df_sum[['doc_id', 'tf_idf_over0.5', 'tf_idf_top50', 'distilled_by_collocation']]
    df_final = pd.merge(df_origin, df_mapping, on='doc_id', how='left')
    
    df_final.to_csv(PATHS['output'], index=False)
    print(f"Done! File saved to: {PATHS['output']}")

if __name__ == "__main__":
    # 실행 시 원하는 컬럼명을 입력하세요.
    # 예: main('cleaned_text') 또는 main('lemmatized_text')
    main(target_col='cleaned_text')
