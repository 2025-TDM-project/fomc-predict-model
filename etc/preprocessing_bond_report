import pandas as pd
bond=pd.read_csv("/content/bond_report.csv")

bond_no_duplicates = bond.drop_duplicates(subset=['date'], keep='first')
df=pd.read_csv('/content/bond_report_no_duplicates.csv')

df.rename(columns={'content':'raw_text'},inplace=True)
df.to_csv('bond_report_new.csv', index=False)


import pandas as pd
import nltk
import string

# 1. 문장 토큰화
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# 2. 품사 태깅 - 언어별 태거 다운로드
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)  # 영어 전용 태거

# 3. 불용어
nltk.download('stopwords', quiet=True)

# 전처리 헬퍼 함수 정의
stop_words_list = nltk.corpus.stopwords.words('english')
punctuations = string.punctuation

def process_sentence(sent):
    """단일 문장을 받아 토큰화, POS 태깅, 불용어 제거를 수행합니다."""
    # 1. 토큰화
    tokens = nltk.word_tokenize(sent)

    # 2. POS 태깅
    pos_tags = [tag for word, tag in nltk.pos_tag(tokens)]

    # 3. 불용어 및 구두점 제거 후 cleaned_text 생성 (소문자 변환 포함)
    cleaned_tokens = [
        t.lower() for t in tokens
        if t.lower() not in stop_words_list and t not in punctuations
    ]

    # 4. cleaned_text (원하시는 JSON 포맷과 일치하도록 문자열로 만듦)
    cleaned_text = ' '.join(cleaned_tokens)

    return {
        'text': sent,
        'tokens': cleaned_tokens,
        'pos_tags': pos_tags,
        'cleaned_text': cleaned_text
    }

def expand_and_process_document_with_meta(row):
    """단일 문서 행을 문장 단위로 분해하고 NLP 처리를 적용하며 문서 메타데이터를 포함합니다."""
    sentences = nltk.sent_tokenize(row['raw_text'])

    sentence_data = []
    for sent_id, sent_text in enumerate(sentences):
        processed = process_sentence(sent_text)

        # 문서 레벨 정보를 각 문장에 포함시켜 반환합니다.
        new_row = {
            'doc_id': row['doc_id'],
            'doc_type': row['doc_type'],
            'date': row['date'],  # date 추가
            'title': row['title'],  # title 추가
            'source': row['source'],
            'language': row['language'],
            'raw_text': row['raw_text'],  # raw_text 추가
            'sent_id': sent_id,
            'text': processed['text'],
            'tokens': processed['tokens'],
            'pos_tags': processed['pos_tags'],
            'cleaned_text': processed['cleaned_text']
        }
        sentence_data.append(new_row)

    return sentence_data

# 1. 문서 레벨 정보 정의
df['doc_id'] = 'bond_report_' + df['date'].astype(str).str.replace('-', '')
df['doc_type'] = 'bond_report'
df['source'] = df['main_path']
df['language'] = 'en'

# 2. 전체 DataFrame에 적용 및 메타데이터 포함
all_sentences_list = df.apply(expand_and_process_document_with_meta, axis=1).tolist()

# 3. 리스트의 리스트를 단일 리스트로 펼치고 새 DataFrame 생성
df_sentences = pd.DataFrame([item for sublist in all_sentences_list for item in sublist])

pd.set_option('display.max_columns', None)
df_sentences.head(100)

df_sentences.to_csv('bond_report_master.csv', index=False)

