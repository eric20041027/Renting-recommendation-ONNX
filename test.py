import requests
from bs4 import BeautifulSoup

def scrape_nchu_rental(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.encoding = 'utf-8' # 確保中文不編碼錯誤
        
        if response.status_code != 200:
            print(f"無法存取網頁，錯誤代碼：{response.status_code}")
            return
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 建立一個字典來存儲資訊
        data = {}
        
        # 該網頁結構通常使用 table 或 div 搭配特定 class
        # 我們抓取所有的表格列 (tr)
        rows = soup.find_all('tr')
        
        for row in rows:
            cols = row.find_all(['th', 'td'])
            if len(cols) >= 2:
                # 去除空白字元與冒號
                key = cols[0].get_text(strip=True).replace('：', '')
                value = cols[1].get_text(strip=True)
                if key:
                    data[key] = value
                    
        # 另外抓取備註或描述部分 (如果有特別的區塊)
        # 有些詳細描述可能在特定的 class 中，例如 'content' 或 'remark'
        description = soup.find('div', class_='detail-content') # 範例 class，視實際狀況調整
        if description:
            data['詳細描述'] = description.get_text(strip=True)

        return data

    except Exception as e:
        print(f"發生錯誤：{e}")
        return None

# 目標 URL
target_url = "https://www.osa.nchu.edu.tw/osa/arm/sys/modules/re/detail.php?rid=608"
result = scrape_nchu_rental(target_url)

if result:
    print("--- 抓取到的租屋資訊 ---")
    for k, v in result.items():
        print(f"{k}: {v}")
else:
    print("未能抓取到資料。")