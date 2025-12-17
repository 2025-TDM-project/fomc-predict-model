!pip install selenium
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import os
from google.colab import drive

# Google Drive 마운트
drive.mount('/content/drive')

# 설정
DRIVE_FOLDER = '/content/drive/MyDrive/scraping_data'  # Drive 저장 경로
CSV_FILE = os.path.join(DRIVE_FOLDER, 'bond_report.csv')
PROGRESS_FILE = os.path.join(DRIVE_FOLDER, 'progress.txt')  # 진행상황 기록
SAVE_INTERVAL = 50  # 50개마다 저장 (자주 저장)
START_ID = 491773
END_ID = 450000
RESTART_INTERVAL = 500  # 500개마다 드라이버 재시작

# 폴더 생성
os.makedirs(DRIVE_FOLDER, exist_ok=True)
print(f"📁 저장 경로: {DRIVE_FOLDER}\n")

# 진행상황 불러오기
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'r') as f:
        last_processed_id = int(f.read().strip())
    start_from = last_processed_id  # 마지막 처리한 ID부터 시작 (이미 -1 처리됨)
    print(f"✅ 진행상황 파일 발견! ID {start_from}부터 이어서 시작합니다.")
else:
    start_from = START_ID
    print(f"✅ 새로 시작합니다. ID {start_from}부터 시작.")

# 기존 데이터 불러오기
if os.path.exists(CSV_FILE):
    existing_df = pd.read_csv(CSV_FILE)
    collected_titles = existing_df['title'].tolist()
    collected_dates = existing_df['date'].tolist()
    collected_contents = existing_df['content'].tolist()
    collected_urls = existing_df['main_path'].tolist()
    print(f"✅ 기존 데이터 {len(existing_df)}개 로드 완료\n")
else:
    collected_titles = []
    collected_dates = []
    collected_contents = []
    collected_urls = []
    print("✅ 새 데이터 수집 시작\n")

def create_driver():
    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
    options.add_argument('--disable-gpu')
    options.page_load_strategy = 'eager'
    return webdriver.Chrome(options=options)

def save_data(current_id):
    """데이터와 진행상황 저장"""
    df_temp = pd.DataFrame({
        'title': collected_titles,
        'date': collected_dates,
        'content': collected_contents,
        'main_path': collected_urls
    })
    df_temp.to_csv(CSV_FILE, index=False, encoding='utf-8-sig')

    # 진행상황 저장
    with open(PROGRESS_FILE, 'w') as f:
        f.write(str(current_id))

    print(f"💾 Drive 저장 완료: {len(df_temp)}개 (현재 ID: {current_id})")

driver = create_driver()
wait = WebDriverWait(driver, 15)
count = 0
success_count = 0

print("🚀 스크래핑 시작...\n")

# 데이터 수집
for i in range(start_from, END_ID, -1):
    url = f'https://tradingeconomics.com/united-states/government-bond-yield/news/{i}'

    # 드라이버 주기적 재시작
    if count > 0 and count % RESTART_INTERVAL == 0:
        print(f"\n🔄 드라이버 재시작 중...")
        driver.quit()
        time.sleep(3)
        driver = create_driver()
        wait = WebDriverWait(driver, 15)

    retries = 3
    success = False

    for attempt in range(retries):
        title = None
        date = None
        content = None

        driver.get(url)
        time.sleep(1.5)

        title_element = wait.until(
            EC.presence_of_element_located((By.XPATH, "//h1[@class='news_title']"))
        )
        title = title_element.text.strip()

        date_element = driver.find_element(By.XPATH, "//div[@class='news_info']//span[1]")
        date = date_element.text.strip()

        content_elements = driver.find_elements(By.XPATH, "//div[@class='news_description']/p")
        if content_elements:
            content = "\n\n".join([e.text.strip() for e in content_elements if e.text.strip()])

        if title and date:
            collected_titles.append(title)
            collected_dates.append(date)
            collected_contents.append(content if content else "")
            collected_urls.append(url)
            success_count += 1
            success = True

            if success_count % 10 == 0:
                print(f"✅ ID {i} 완료 (총 {success_count}개)")
            break

        elif attempt < retries - 1:
            time.sleep(2)

    if not success:
        print(f"❌ ID {i} 스킵")

    count += 1

    # 주기적 저장 (Drive에 저장)
    if success_count > 0 and success_count % SAVE_INTERVAL == 0:
        save_data(i)

    # 진행상황 표시
    if count % 100 == 0:
        print(f"\n📊 진행: {count}회 시도, {success_count}개 수집, 현재 ID: {i}\n")

# 최종 저장
driver.quit()
save_data(END_ID)

print("\n" + "="*60)
print("✅ 스크래핑 완료!")
print("="*60)
print(f"총 수집: {len(collected_titles)}개")
print(f"저장 위치: {CSV_FILE}")
print(f"진행 기록: {PROGRESS_FILE}")

# 최종 결과
df_final = pd.read_csv(CSV_FILE)
print(f"\n최종 데이터프레임: {len(df_final)}행")
print(df_final.head())
