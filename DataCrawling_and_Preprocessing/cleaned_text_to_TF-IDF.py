import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# 1. 설정 및 경로 (Configuration)
# ==========================================
PATHS = {
    'bond': '/content/bond_report_master.csv',
    'news': '/content/news_posclean_dirty.csv',
    'fomc': '/content/FOMC_full_sentence_level_with_sections_3.csv',
    'collocation_pickle': '/content/collocation_list.pkl',
    'n_grams': [
        '/content/bigram_collocations.csv',
        '/content/trigram_collocations.csv',
        '/content/fourgram_collocations.csv',
        '/content/fivegram_collocations.csv'
    ],
    'output': 'df_master_TM_proj.csv'
}

# ==========================================
# 2. 데이터 로드 및 병합 (Data Loading)
# ==========================================
def load_and_merge_documents():
    """본문 데이터(Bond, News, FOMC)를 로드하고 하나로 병합합니다."""
    cols = ['doc_id', 'cleaned_text']
    
    df_bond = pd.read_csv(PATHS['bond'], usecols=cols)
    df_news = pd.read_csv(PATHS['news'], usecols=cols)
    df_fomc = pd.read_csv(PATHS['fomc'], usecols=cols)

    df_merged = pd.concat([df_bond, df_news, df_fomc], axis=0, ignore_index=True)
    df_merged['cleaned_text'] = df_merged['cleaned_text'].fillna('')
    
    return df_merged

def load_collocation_set():
    """Pickle 파일에서 연어(Collocation) 리스트를 로드하여 set으로 반환합니다."""
    # (참고) N-gram CSV 로드 로직이 원본에 있었으나, 
    # 실제 필터링에는 pickle 파일이 사용되므로 pickle 로드만 수행합니다.
    with open(PATHS['collocation_pickle'], 'rb') as f:
        collocation_text_list = pickle.load(f)
    return set(collocation_text_list)

# ==========================================
# 3. 텍스트 전처리 (Preprocessing)
# ==========================================
def preprocess_documents(df):
    """문서를 doc_id 기준으로 그룹화하고 텍스트를 재구성합니다."""
    # doc_id별로 텍스트 합치기
    df_sum = df.groupby('doc_id')['cleaned_text'].agg(lambda x: ' '.join(x)).reset_index()
    df_sum.rename(columns={'cleaned_text': 'full_document'}, inplace=True)
    
    # 중복 단어 제거 (Set 사용)
    df_sum['full_document'] = df_sum['full_document'].apply(lambda x: ', '.join(set(x.split())))
    
    return df_sum

def filter_by_collocation(text, collocation_set):
    """텍스트에서 Collocation Set에 포함된 단어만 추출합니다."""
    words = str(text).split(', ')
    # 리스트 컴프리헨션으로 필터링
    filtered_words = [word for word in words if word in collocation_set]
    return ' '.join(filtered_words)

# ==========================================
# 4. TF-IDF 분석 (Analysis)
# ==========================================
def get_keywords_over_threshold(doc_idx, tfidf_matrix, feature_names, min_score=0.5):
    """임계값(min_score) 이상의 키워드를 추출합니다."""
    vector = tfidf_matrix[doc_idx].toarray().flatten()
    
    # 임계값 이상인 인덱스 추출
    filtered_indices = np.where(vector >= min_score)[0]
    
    if len(filtered_indices) == 0:
        return ""
    
    # 점수 기준 내림차순 정렬
    sorted_indices = filtered_indices[np.argsort(vector[filtered_indices])[::-1]]
    
    result = [f"{feature_names[i]}: {vector[i]:.3f}" for i in sorted_indices]
    return ', '.join(result)

def get_top_n_keywords(doc_idx, tfidf_matrix, feature_names, top_n=50):
    """상위 N개의 키워드를 추출합니다."""
    vector = tfidf_matrix[doc_idx].toarray().flatten()
    
    # 상위 N개 인덱스 추출 (내림차순)
    top_indices = vector.argsort()[-top_n:][::-1]
    
    result = [f"{feature_names[i]}: {vector[i]:.3f}" for i in top_indices]
    return ', '.join(result)

# ==========================================
# 5. 메인 실행 흐름 (Main Execution)
# ==========================================
def main():
    print("1. Loading Documents...")
    df_origin = load_and_merge_documents()
    
    print("2. Grouping Documents by ID...")
    df_sum = preprocess_documents(df_origin)
    
    print("3. Loading Collocations & Filtering...")
    collocation_set = load_collocation_set()
    
    # 주의: 원본 코드의 'lemmatized_text'는 존재하지 않아 'full_document'로 대체했습니다.
    df_sum['distilled_by_collocation'] = df_sum['full_document'].apply(
        lambda x: filter_by_collocation(x, collocation_set)
    )

    print("4. Running TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_sum['distilled_by_collocation'])
    feature_names = vectorizer.get_feature_names_out()

    print("5. Extracting Keywords...")
    # (1) Threshold 0.5 이상
    df_sum['tf_idf_over0.5'] = [
        get_keywords_over_threshold(i, tfidf_matrix, feature_names, min_score=0.5) 
        for i in range(len(df_sum))
    ]
    
    # (2) Top 50 키워드
    df_sum['tf_idf_top50'] = [
        get_top_n_keywords(i, tfidf_matrix, feature_names, top_n=50) 
        for i in range(len(df_sum))
    ]

    print("6. Merging Results & Saving...")
    # 필요한 컬럼만 선택하여 병합
    df_mapping = df_sum[['doc_id', 'tf_idf_over0.5', 'tf_idf_top50', 'distilled_by_collocation']]
    
    # 원본 데이터(문장 단위)에 문서 단위 키워드 정보를 매핑
    df_final = pd.merge(df_origin, df_mapping, on='doc_id', how='left')
    
    df_final.to_csv(PATHS['output'], index=False)
    print(f"Done! File saved to: {PATHS['output']}")

if __name__ == "__main__":
    main()import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# 1. 설정 및 경로 (Configuration)
# ==========================================
PATHS = {
    'bond': '/content/bond_report_master.csv',
    'news': '/content/news_posclean_dirty.csv',
    'fomc': '/content/FOMC_full_sentence_level_with_sections_3.csv',
    'collocation_pickle': '/content/collocation_list.pkl',
    'n_grams': [
        '/content/bigram_collocations.csv',
        '/content/trigram_collocations.csv',
        '/content/fourgram_collocations.csv',
        '/content/fivegram_collocations.csv'
    ],
    'output': 'df_master_TM_proj.csv'
}

# ==========================================
# 2. 데이터 로드 및 병합 (Data Loading)
# ==========================================
def load_and_merge_documents():
    """본문 데이터(Bond, News, FOMC)를 로드하고 하나로 병합합니다."""
    cols = ['doc_id', 'cleaned_text']
    
    df_bond = pd.read_csv(PATHS['bond'], usecols=cols)
    df_news = pd.read_csv(PATHS['news'], usecols=cols)
    df_fomc = pd.read_csv(PATHS['fomc'], usecols=cols)

    df_merged = pd.concat([df_bond, df_news, df_fomc], axis=0, ignore_index=True)
    df_merged['cleaned_text'] = df_merged['cleaned_text'].fillna('')
    
    return df_merged

def load_collocation_set():
    """Pickle 파일에서 연어(Collocation) 리스트를 로드하여 set으로 반환합니다."""
    # (참고) N-gram CSV 로드 로직이 원본에 있었으나, 
    # 실제 필터링에는 pickle 파일이 사용되므로 pickle 로드만 수행합니다.
    with open(PATHS['collocation_pickle'], 'rb') as f:
        collocation_text_list = pickle.load(f)
    return set(collocation_text_list)

# ==========================================
# 3. 텍스트 전처리 (Preprocessing)
# ==========================================
def preprocess_documents(df):
    """문서를 doc_id 기준으로 그룹화하고 텍스트를 재구성합니다."""
    # doc_id별로 텍스트 합치기
    df_sum = df.groupby('doc_id')['cleaned_text'].agg(lambda x: ' '.join(x)).reset_index()
    df_sum.rename(columns={'cleaned_text': 'full_document'}, inplace=True)
    
    # 중복 단어 제거 (Set 사용)
    df_sum['full_document'] = df_sum['full_document'].apply(lambda x: ', '.join(set(x.split())))
    
    return df_sum

def filter_by_collocation(text, collocation_set):
    """텍스트에서 Collocation Set에 포함된 단어만 추출합니다."""
    words = str(text).split(', ')
    # 리스트 컴프리헨션으로 필터링
    filtered_words = [word for word in words if word in collocation_set]
    return ' '.join(filtered_words)

# ==========================================
# 4. TF-IDF 분석 (Analysis)
# ==========================================
def get_keywords_over_threshold(doc_idx, tfidf_matrix, feature_names, min_score=0.5):
    """임계값(min_score) 이상의 키워드를 추출합니다."""
    vector = tfidf_matrix[doc_idx].toarray().flatten()
    
    # 임계값 이상인 인덱스 추출
    filtered_indices = np.where(vector >= min_score)[0]
    
    if len(filtered_indices) == 0:
        return ""
    
    # 점수 기준 내림차순 정렬
    sorted_indices = filtered_indices[np.argsort(vector[filtered_indices])[::-1]]
    
    result = [f"{feature_names[i]}: {vector[i]:.3f}" for i in sorted_indices]
    return ', '.join(result)

def get_top_n_keywords(doc_idx, tfidf_matrix, feature_names, top_n=50):
    """상위 N개의 키워드를 추출합니다."""
    vector = tfidf_matrix[doc_idx].toarray().flatten()
    
    # 상위 N개 인덱스 추출 (내림차순)
    top_indices = vector.argsort()[-top_n:][::-1]
    
    result = [f"{feature_names[i]}: {vector[i]:.3f}" for i in top_indices]
    return ', '.join(result)

# ==========================================
# 5. 메인 실행 흐름 (Main Execution)
# ==========================================
def main():
    print("1. Loading Documents...")
    df_origin = load_and_merge_documents()
    
    print("2. Grouping Documents by ID...")
    df_sum = preprocess_documents(df_origin)
    
    print("3. Loading Collocations & Filtering...")
    collocation_set = load_collocation_set()
    
    # 주의: 원본 코드의 'lemmatized_text'는 존재하지 않아 'full_document'로 대체했습니다.
    df_sum['distilled_by_collocation'] = df_sum['full_document'].apply(
        lambda x: filter_by_collocation(x, collocation_set)
    )

    print("4. Running TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_sum['distilled_by_collocation'])
    feature_names = vectorizer.get_feature_names_out()

    print("5. Extracting Keywords...")
    # (1) Threshold 0.5 이상
    df_sum['tf_idf_over0.5'] = [
        get_keywords_over_threshold(i, tfidf_matrix, feature_names, min_score=0.5) 
        for i in range(len(df_sum))
    ]
    
    # (2) Top 50 키워드
    df_sum['tf_idf_top50'] = [
        get_top_n_keywords(i, tfidf_matrix, feature_names, top_n=50) 
        for i in range(len(df_sum))
    ]

    print("6. Merging Results & Saving...")
    # 필요한 컬럼만 선택하여 병합
    df_mapping = df_sum[['doc_id', 'tf_idf_over0.5', 'tf_idf_top50', 'distilled_by_collocation']]
    
    # 원본 데이터(문장 단위)에 문서 단위 키워드 정보를 매핑
    df_final = pd.merge(df_origin, df_mapping, on='doc_id', how='left')
    
    df_final.to_csv(PATHS['output'], index=False)
    print(f"Done! File saved to: {PATHS['output']}")

if __name__ == "__main__":
    main()
