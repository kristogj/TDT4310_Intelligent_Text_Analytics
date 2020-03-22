from bs4 import BeautifulSoup
import requests

url = "https://www.ntnu.edu/vacancies"

# Get html page
r = requests.get(url=url)
html_doc = r.text

soup = BeautifulSoup(html_doc, 'html.parser')

# Get all h3 elements with jobs
mydivs = soup.find("div", {"class": "vacancies"})

# Extract jobs into a list
jobs = mydivs.findAll("h3")

# Number of jobs
print("Number of jobs:")
print("\t" + str(len(jobs)))

# Print all titles
deadlines = []
print("Job titles:")
for job in jobs:
    title = job.find("a").getText()
    print("\t" + title)
    deadline = title.split("Søknadsfrist:")[-1]
    deadlines.append(deadline)

print("Deadlines: ")
for deadline in deadlines:
   print("\t" + deadline)