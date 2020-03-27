import time
import glob
from utils import crawl_meta
from nltk import FreqDist


def get_meta_data(crawl=False, crawl_reviews=False):
    # Get the meta data
    if crawl:

        # Uncomment this if you want to crawl data from scratch
        meta_list = crawl_meta(
            meta_hdf5=None,
            write_meta_name='data_{}.hdf5'.format(time.strftime("%Y%m%d%H%M%S")),
            crawl_review=crawl_reviews)
        datatime = time.strftime(
            '%m/%d/%Y %I:%M:%S %p', time.strftime("%Y%m%d%H%M%S"))
    else:
        # Uncomment this if you want to load the previously stored data file
        meta_hdf5s = glob.glob("*.hdf5")
        meta_hdf5 = sorted(meta_hdf5s)[-1]  # the most recent data
        print("Loading {} ...".format(meta_hdf5))
        meta_list = crawl_meta(meta_hdf5=meta_hdf5)
        datatime = time.strftime(
            '%m/%d/%Y %I:%M:%S %p',
            time.strptime(meta_hdf5.split('_')[-1].split('.')[0], '%Y%m%d%H%M%S'))
    print(datatime)
    return meta_list


def acceptance_rate(meta_list):
    print("Task 4a")
    num_withdrawn = len([m for m in meta_list if m.withdrawn or m.desk_reject])
    total = len(meta_list)
    acceptance_rate = (total - num_withdrawn) / total
    print("Number of submissions: {}".format(total))
    print("Number of withdrawns / desk reject submissions: {}".format(num_withdrawn))
    print("Acceptance rate: {}\n".format(acceptance_rate))


def attribute_freq(meta_list, attribute="keyword"):
    if attribute == "keyword":
        attributes = [keyword for paper in meta_list for keyword in paper.keyword]
    elif attribute == "author":
        attributes = [keyword for paper in meta_list for keyword in paper.author]
    else:
        raise AttributeError("Missing attribute")

    freq_dist = FreqDist(attributes)

    # Remove the empty key
    freq_dist.__delitem__('')

    # Sort by frequency and report top 10
    counts = sorted(list(freq_dist.items()), key=lambda x: x[1], reverse=True)[:10]
    for i, (att, count) in enumerate(counts):
        print("{}. {} ({})".format(i, att, count))
    print()


if __name__ == '__main__':
    print("Lab 4 Exercise 4")
    CRAWL_DATA = False
    CRAWL_REVIEW = False

    meta_list = get_meta_data(crawl=CRAWL_DATA, crawl_reviews=CRAWL_REVIEW)

    # Task 4a - Acceptance rate
    acceptance_rate(meta_list)

    # Task 4b - top 10 frequent keywords
    print("Task 4b - top 10 frequent keywords")
    attribute_freq(meta_list, attribute="keyword")

    # Task 4c - top 10 frequent authors
    print("Task 4c - Top 10 frequent authors")
    attribute_freq(meta_list, attribute="author")
