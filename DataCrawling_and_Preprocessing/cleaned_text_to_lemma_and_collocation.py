import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import numpy as np
import re
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

#import df
df=pd.read_csv(r'C:\TMD_Project\df_master_TM_proj.csv')

#Lemmatizing

# 2. Lemmatizer 초기화
lemmatizer = WordNetLemmatizer()

# 3. POS tag를 WordNet tag로 변환하는 함수
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# 4. Lemmatization 함수
def lemmatize_text(text):
    if pd.isna(text):
        return ''

    words = text.split()
    pos_tags = nltk.pos_tag(words)
    lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(pos))
                  for word, pos in pos_tags]
    return ' '.join(lemmatized)

# 5. 데이터프레임에 적용
df['lemmatized_text'] = df['cleaned_text'].apply(lemmatize_text)


#Collocation
#연어 모음 불러오기
import pickle

file_path=r'C:\TMD_Project\collocation_list.pkl'
# 'rb': 읽기 모드 (Read Binary)
with open(file_path, 'rb') as f:
    collocation_text_list = pickle.load(f)
collocation_set = set(collocation_text_list)

def collocation_words(col):
    words = str(col).split(', ')
    return ' '.join([word for word in words if word in collocation_set])

df['distilled_by_collocation']=df['lemmatized_text'].apply(collocation_words)


#Sentimental Analysis
# --- 1. 데이터 전처리 (단어 추출 및 토큰화) ---

# df['cleaned_text'] 열이 'time inflation year'와 같은 문자열이라고 가정합니다.
# 이를 공백 기준으로 나누어 단어 리스트로 변환합니다.
def tokenize_string(text):
    return str(text).split()

# 새로운 열에 토큰화된 리스트 저장 (이 열을 학습에 사용합니다)
df['tokenized_collocation'] = df['cleaned_text'].apply(tokenize_string)

# --- 2. Word2Vec 모델 학습 ---

# Word2Vec 입력: 단어 리스트의 리스트 [[w1, w2], [w3, w4], ...]
training_data = df['tokenized_collocation'].tolist()

print(f"학습 데이터 예시 (첫 2개 행): {training_data[:2]}")
#

# 모델 학습
model = Word2Vec(sentences=training_data,
                 vector_size=100,
                 window=10,      # 한 행의 단어들이 서로 관련있다고 가정
                 min_count=1,
                 workers=4,
                 sg=1)           # sg=1: Skip-gram (의미론적 관계 파악에 유리)

# --- 3. 씨앗 단어 정의 및 필터링 ---
# 매파 (Hawkish) - 긴축 정책 선호
seeds_hawk= [
    'inflation', 'tightening', 'rate', 'hike', 'increase', 'restrict',
    'hawkish', 'aggressive', 'curb', 'combat', 'control', 'restrain',
    'normalize', 'normalization', 'contractionary', 'raising', 'higher',
    'cool', 'cooling', 'brake', 'slowdown', 'decelerate',
    'overheat', 'overheating', 'pressure', 'concern', 'worry',
    'vigilant', 'cautious', 'prudent', 'tough', 'firm',
    'withdraw', 'withdrawal', 'taper', 'tapering', 'reduce',
    'unwind', 'unwinding', 'drain', 'shrink', 'balance', 'sheet', 'reduction',
    'quantitative', 'tightening', 'terminal', 'rate', 'neutral', 'rate',
    'upward', 'lift', 'elevated', 'persistent', 'sticky',

    # 🌟 추가된 단어
    'front',' load', 'expeditious', 'unwavering', 'decisive', 'pre','emptive', 'successive', 'above', 'neutral',
    'underlying', 'inflation', 'price','stability', 'inflationary', 'pressures', 'wage', 'growth', 'second','round', 'effects', 'tight', 'labor', 'market',
    'vigilance', 'resolve', 'intransigent', 'intractable', 'commitment', 'credibility',
    'selling','assets', 'runoff', 'draining', 'reserves', 'real rate',
    'necessary', 'evil', 'painful', 'adjustment', 'imperative', 'non','negotiable'
]

# 비둘기파 (Dovish) - 완화 정책 선호
seeds_dove= [
    'growth', 'cut', 'easing', 'stimulus', 'reduction', 'lower',
    'dovish', 'accommodative', 'support', 'boost', 'encourage', 'promote',
    'expansionary', 'lowering', 'decrease', 'ease', 'relax',
    'patient', 'gradual', 'cautious', 'measured', 'pause',
    'hold', 'steady', 'maintain', 'wait', 'monitor',
    'recovery', 'employment', 'jobs', 'unemployment', 'labor',
    'weak', 'weakness', 'soft', 'softness', 'slow',
    'dovish', 'inject', 'injection', 'liquidity', 'provide',
    'quantitative easing', 'asset purchase', 'accommodation',
    'downward', 'decline', 'subdued', 'moderate', 'muted',
    'flexible', 'data dependent', 'stabilize', 'stability',

    # 🌟 추가된 단어
    'supportive', 'extraordinary', 'measures', 'forward', 'guidance', 'open','ended', 'reinvest',
    'maximum', 'employment', 'slack', 'underutilized', 'resources', 'headwinds', 'transitory', 'symmetric','target',
    'flexibility', 'wait','and','see', 'patiently', 'adjust', 'contingent',
    'downside', 'risks', 'uncertain', 'outlook', 'fragile', 'recovery', 'output', 'gap',
    'benign', 'contained', 'manageable', 'well','anchored'
]

# 모델 단어 사전에 있는 씨앗 단어만 유효하게 남김
valid_hawk = [w for w in seeds_hawk if w in model.wv]
valid_dove = [w for w in seeds_dove if w in model.wv]

print(f"유효한 매파 씨앗: {valid_hawk}")
print(f"유효한 비둘기파 씨앗: {valid_dove}")

# --- 4. 극성 분석 함수 정의 ---

def get_polarity_score(target_word, model, hawks, doves):
    """
    단어의 극성 점수를 계산 (매파 유사도 - 비둘기파 유사도)
    """
    if target_word not in model.wv or not hawks or not doves:
        return 0.0

    # 매파 그룹과의 평균 유사도
    sim_hawk = np.mean([model.wv.similarity(target_word, h) for h in hawks])
    # 비둘기파 그룹과의 평균 유사도
    sim_dove = np.mean([model.wv.similarity(target_word, d) for d in doves])

    # 극성 점수 (양수: 매파적, 음수: 비둘기파적)
    return sim_hawk - sim_dove

def analyze_row_polarity(word_list):
    """
    한 행(단어 리스트)에 있는 모든 단어의 극성을 분석하여 딕셔너리로 반환
    """
    row_results = {}
    for word in word_list:
        score = get_polarity_score(word, model, valid_hawk, valid_dove)
        # 0이 아닌 유의미한 점수가 있는 경우만 저장
        if score != 0:
            row_results[word] = round(score, 4)
    return row_results

# --- 5. 결과 적용 (새로운 컬럼 생성) ---

# 토큰화된 리스트 열을 사용하여 분석 수행
df['polarity_analysis_result'] = df['tokenized_collocation'].apply(analyze_row_polarity)

# --- 결과 확인 ---
pd.set_option('display.max_colwidth', None) # 내용 잘림 방지
print("\n=== 분석 결과 (상위 5행) ===")
print(df[['cleaned_text', 'polarity_analysis_result']].head())

# (선택) 전체 행에 대한 평균 매파/비둘기파 성향 점수 계산 예시
def calculate_aggregate_score(result_dict):
    """
    문서에 포함된 모든 단어의 극성 점수 평균을 계산하여 문서 전체 성향 점수를 산출
    """
    if not result_dict:
        return 0.0
    return np.mean(list(result_dict.values()))

df['doc_sentiment_score'] = df['polarity_analysis_result'].apply(calculate_aggregate_score)

print("\n=== 문서 전체 성향 점수 (상위 5행) ===")
print(df[['cleaned_text', 'doc_sentiment_score']].head())

df.to_csv(r'C:\TMD_Project\df_TM_Lem_Col_TF.csv', index=False)
