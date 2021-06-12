from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = webdriver.ChromeOptions()
options.add_argument("user-data-dir=C:\\Users\\18364\\AppData\\Local\\Google\\Chrome\\User Data\\Default")
driver = webdriver.Chrome(chrome_options=options)
driver.get("https://www.coursera.org/learn/neural-networks-deep-learning/programming/isoAV/python-basics-with-numpy/lab")

# driver = webdriver.Chrome()
# driver.get('https://www.coursera.org/learn/neural-networks-deep-learning/programming/isoAV/python-basics-with-numpy/lab')

