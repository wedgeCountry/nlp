from bs4 import BeautifulSoup
from multiprocessing.dummy import Pool
import os
import shutil
import time
import requests
import re
import json
import glob

UNKNOWN_FORMAT = 0
APPELLATION_FORMAT_0 = 1
APPELLATION_FORMAT_1 = 2
APPELLATION_FORMAT_2 = 3
HEADERS = {
    'user-agent': ('Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/48.0.2564.109 Safari/537.36')
}
    
class Scraper(object):
    """
    Scraper code taken from 
    https://github.com/zackthoutt/wine-deep-learning/
    """
    def __init__(self, num_jobs=1):
        object.__init__(self)
        self.session = requests.Session()
        self.set_num_jobs(num_jobs)
        self.start_time = time.time()
        
    def set_num_jobs(self, num_jobs):
        self.num_jobs = num_jobs
        if self.num_jobs > 1:
            self.multiprocessing = True
            self.worker_pool = Pool(self.num_jobs)
        else:
            self.multiprocessing = False

    def scrape_pages_for_links(self, pages):
        """ get a list of links for further scraping """
        links = []
        for page in pages:
            links.extend(self.scrape_page_for_links(page))
        return links
        
    def scrape_pages(self, pages):
        """ scrape a list of links for specific info which can be saved as json, e.g. """
        if self.multiprocessing:
            records = self.worker_pool.map(self.scrape_page, pages)
            self.worker_pool.terminate()
            self.worker_pool.join()
        else:
            records = []
            for page in pages:
                records.extend(self.scrape_page(page))
        return records
        
    def scrape_page(self, page_url):
        raise NotImplementedError("derive from this class!")

    def scrape_page_for_links(self, page_url):
        raise NotImplementedError("derive from this class!")   
        
    def get_response_content(self, page_url, retry_count=0):
        try:
            response = self.session.get(page_url, headers=HEADERS)
        except:
            retry_count += 1
            if retry_count <= 3:
                self.session = requests.Session()
                return self.get_response_content(page_url, retry_count)
            else:
                raise
        return response.content

    
class CategoryScraper(Scraper):
    """
    Scrape the categories of the page
    """
    def scrape_page(self, page_url):
        print("CategoryScraper", page_url, "\n")
        content = self.get_response_content(page_url, 0)
        soup = BeautifulSoup(content, 'html.parser')
        
        main_container = soup.find("div", attrs={"id": "entries-selector"})
        #print(main_container)
        return [li["value"] for li in main_container.find_all("option")]

class LinkScraper(Scraper):
    """
    For a page scrape the links of the words
    """
    def scrape_page(self, page_url):
        print("LinkScraper", page_url, "\n")
        content = self.get_response_content(page_url, 0)
        soup = BeautifulSoup(content, 'html.parser')
        
        main_container = soup.find("div", id="entrylist1")
        return[li.a["href"] for li in main_container.find_all("li")]   

class WordFunctionScraper(Scraper):
    """
    On the page of a word, scrape the word function
    """
    def scrape_page(self, page_url):
        print("WordFunctionScraper", page_url, "\n")
        content = self.get_response_content(page_url, 0)
        #print(content)
        soup = BeautifulSoup(content, 'html.parser')
        word = ''
        try:
            main = soup.find("div", attrs={"class": "webtop-g"})
            word = main.find("h2").text
            function = main.find("span", attrs = {"class": "pos"}).text
        except:
            if word:
                return {"word": ''}
            else:
                #print "page did not have a word: %s" %page_url
                #print main
                return {}
        return {word: function}
        

if __name__ == '__main__':
    
    # save the words here
    jsonfile = "words.txt"
    
    # load the words we already got
    if not os.path.isfile(jsonfile):
        word_dict = {}
    else:
        with open(jsonfile, "r") as jfile:
            word_dict = json.load(jfile)
    
    # make unique
    for key in word_dict.keys():
        word_dict[key] = list(set([m.strip() for m in word_dict[key] if m.strip()]))
    
    # go to 3000 word page on Oxford dictionary and get all categories
    start_page = "https://www.oxfordlearnersdictionaries.com/wordlist/english/oxford3000/"
    links = CategoryScraper(num_jobs=1).scrape_pages([start_page])
    links.append(start_page) # add A-B
    
    # for each category get all links of the pages
    scraper = LinkScraper(num_jobs=1)
    word_links = []
    for link in links:
        if not link:
            continue
        print("Scraping %s" %link, len(word_links))
        for i in range(1,21):
            page = "%s?page=%d" % (link, i)
            further_links = scraper.scrape_pages([page])
            if not further_links:
                break
            else: 
                word_links.extend(further_links)
    
    # if multiple grammaticalic meanings get the first max_num_meanings
    max_num_meanings = 3 
    pages = []
    for link in word_links:
        word = link.split("/")[-1]
        # e.g. https://www.oxfordlearnersdictionaries.com/definition/english/beat_1 has got two other meanings
        if word[-2] == "_":
            # some phrases are composed
            word = word[:-2].replace("-", " ")
            # names are capitalized
            cap_word = "%s%s" % (word[0].upper(), word[1:])
            if word not in word_dict and cap_word in word_dict:
                word = cap_word
            # check if all meanings are already found
            min_num_meanings = 0
            if word in word_dict: 
                min_num_meanings = len(word_dict[word])
            for i in range(min_num_meanings + 1, max_num_meanings + 1):
                pages.append("%s%d" % (link[:-1], i))
        elif word in word_dict:
            continue
        else:
            # just take the one and only meaning
            pages.append(link)
    pages.sort()
    
    # scrape the word function
    scraper = WordFunctionScraper(num_jobs=1)
    scraper.set_num_jobs(16)
    words = scraper.scrape_pages(pages)
    
    # save features in dict
    for word in words:
        for key, value in word.items():
            if key not in word_dict:
                word_dict[key] = [value]
            else:
                word_dict[key].append(value)
    
    for key in word_dict.keys():
        word_dict[key] = list(set([m.strip() for m in word_dict[key] if m.strip()]))
        
    with open(jsonfile, "w+") as jfile:
        json.dump(word_dict, jfile, sort_keys=True, indent=4)
    print("DONE")
    

    # word2vec 
    # 
    # https://www.quora.com/How-does-word2vec-work-Can-someone-walk-through-a-specific-example
