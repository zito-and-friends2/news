import requests as rq
from bs4 import BeautifulSoup
import json
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from dotenv import load_dotenv
import os
from summarizeNews import summarize_article_text
load_dotenv()


API_URL = os.getenv("API_URL")
ID = os.getenv("ID")
SECRET_KEY = os.getenv("SECRET_KEY")
KEYWORD_URL = os.getenv("KEYWORD_URL")


NEWS_URL = "https://openapi.naver.com/v1/search/news.json?query="
NEWS_PARAM = "&display=20&start=1&sort=sim"



title=""
link=""
url=""
categories=""
pubdate=""
image=""
contents=""


def main():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(options=options)
    

    try:
        res = rq.get(KEYWORD_URL).text
        list = json.loads(res)
        headers = {"X-Naver-Client-Id": ID, "X-Naver-Client-Secret": SECRET_KEY}

        for i in range(0, 10):
            keyword = list["top10"][i]["keyword"]
            news = rq.get(NEWS_URL + keyword + NEWS_PARAM, headers=headers).text
         

            
            news = json.loads(news)

            
            if "items" not in news:
                print(f"경고: '{keyword}'에 대한 'items' 키가 없습니다. 해당 키워드를 건너뜁니다.")
                continue 

            print(keyword)
            for j in range(0, 10):
              
                if "naver.com" in news["items"][j]["link"]:
                  
                    title = news["items"][j]["title"]
                    clean_title = re.sub(r'<.*?>', '', title)


                    clean_title = re.sub(r'&[a-zA-Z0-9#]+;', '', clean_title) 

                    link = news["items"][j]["link"]
                    originallink = news["items"][j]["originallink"]
                    url = link
                    categories = get_news_categories(url)
                    print(categories)
                    pubdate = news["items"][j]["pubDate"]
                    categories = categories[0]
                    print("제목 : " + clean_title)
                    print("주소 : " + originallink)
                    print("날짜 : " + pubdate)
                    

                    if "entertain" in url or "sports" in url:
                        article, image = get_news_article_text_selenium(url, driver)
                        contents = article
                        image = image
                        print(article)
                        print("이미지 : " + image)
                    else:
                        print(getContents(url))
                        contents = getContents(url)
                        print("이미지 : " + get_news_img(url))
                        image = get_news_img(url)
                    print("===========================================")





                    summary_result = summarize_article_text(contents,title=clean_title)
                    summary_sentence = "\n".join(summary_result['summary'])
                    summary_keywords = ",".join(summary_result['keywords'])
                    data = {
                                "title": clean_title,
                                "pub_date": str(pubdate),
                                "content": summary_sentence,
                                "org_link": originallink,
                                "category": categories,
                                "img_url" : image,
                                "keyword" : keyword,
                                "hashtag" : summary_keywords

                    }

                    headers2 = {
                                "Content-Type": "application/json"
                    }

                    response = rq.post(API_URL, headers=headers2, json=data)
                    print(response.text)
                    break

                    


    finally:
        driver.quit()


def get_news_img(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = rq.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    img_tag = soup.find("img", {"id": "img1"})
    if img_tag:
        image_url = img_tag.get("data-src")
        return image_url if image_url else "이미지 없음"
    return "이미지 없음"


def get_news_categories(url):
    try:
        if "entertain" in url:
            return ["연예"]

        if "sports" in url:
            return ["스포츠"]

        headers = {"User-Agent": "Mozilla/5.0"}
        res = rq.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")

        categories = soup.select("#contents > div.media_end_categorize > a > em")
        return [cat.text.strip() for cat in categories] if categories else ["없음"]

    except Exception as e:
        print(f"[카테고리 추출 오류] {e}")
        return ["오류"]

def getContents(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = rq.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    contents = soup.select("#dic_area")
    cleaned_result = re.sub(r'<.*?>', '', str(contents))
    cleaned_result = re.sub(r'\s+', ' ', cleaned_result)
    cleaned_result = cleaned_result.strip()
    return cleaned_result

def get_news_article_text_selenium(url, driver):
    try:
        driver.get(url)
        driver.implicitly_wait(3)

        contents = driver.find_elements(By.CSS_SELECTOR, "#comp_news_article")
        raw_text = "\n".join([c.get_attribute("innerHTML") for c in contents])
        cleaned_result = re.sub(r'<.*?>', '', raw_text)
        cleaned_result = re.sub(r'\s+', ' ', cleaned_result).strip()

        span_tag = driver.find_element(By.CSS_SELECTOR, "span.NewsEndMain_image_wrap__djL-o")
        img_tag_in_span = span_tag.find_element(By.TAG_NAME, "img")
        image_url_span = img_tag_in_span.get_attribute("src")

        return [cleaned_result, image_url_span]

    except Exception as e:
        print(f"[ 본문 오류] {e}")
        return False




if __name__ == "__main__":
    main()