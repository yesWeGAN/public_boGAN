import requests
import os
import sys
import pickle
import glob
import argparse
import time
from tqdm import tqdm


class AnonymPageScraper:
    """public version of a simple Page crawler. anonymized to protect website"""
    def __init__(self,
                 storage_dir,
                 baseurl="https://www.ANONYM.com/ANONYM/",
                 header={'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:97.0) Gecko/20100101 Firefox/97.0'},
                 time_delay=1,
                 verbose=False,
                 ckpt_version=1):
        self.STORAGE = storage_dir
        self.HEADER = header
        self.BASEURL = baseurl
        self.TIME_DELAY = time_delay
        self.verbose = verbose
        self.response_hash = {}
        self.ckpt_version = ckpt_version

    def say_hello(self):
        """function call to inform about crawling progress at the beginning (CMD-LINE)"""
        print("Hi, I am ANONYMPageScraper version", self.ckpt_version)
        print("I will resume from", max(self.response_hash.keys()))
        print("Time delay currently set to:", self.TIME_DELAY)
        print("Storing under", self.STORAGE)
        print("Let's go!")

    def set_verbose(self, boolean):
        self.verbose = boolean

    def get_url_from_index(self, index):
        """construct a valid url from page-index"""
        return self.BASEURL + str(index)

    def make_dir_to_store(self, index):
        if not os.path.isdir(os.path.join(self.STORAGE, str(index))):
            os.mkdir(os.path.join(self.STORAGE, str(index)))
        return os.path.join(self.STORAGE, str(index))

    def get_page_response(self, index):
        """access the page"""
        response = requests.get(url=self.get_url_from_index(index), headers=self.HEADER)

        if response.status_code == 200:
            return response

        else:
            self.response_hash[index] = response.status_code
            return None

    def store_page_response(self, response, path, index):
        """stores raw html file for later-on parsing and downloading"""

        filename = "tour_" + str(index) + ".html"

        if self.verbose:
            print("Storing under filepath", os.path.join(path, filename))

        with open(os.path.join(path, filename), 'w', encoding='utf8') as htmlfile:
            htmlfile.write(response.text)

        pass

    def search_index_range(self, startindex, endindex):
        """download all media within specific XML-range"""

        for index in range(startindex, endindex):

            if not index in self.response_hash.keys():

                response = self.get_page_response(index)
                if response is not None:
                    store_path = self.make_dir_to_store(index)
                    self.store_page_response(response, store_path, index)
                    self.response_hash[index] = "200"
                else:
                    if self.verbose:
                        print("None found for index", index)

            elif self.verbose:
                print("already checked index:", index)

    def store_self_checkpoint(self):
        """allow class to store its state for later-on restart of execution"""
        self.ckpt_version += 1
        pickle.dump(self, open(os.path.join(self.STORAGE, "komoot_ckpt_" + str(self.ckpt_version) + ".pkl"), 'wb'))
        pass

    def continue_index_search(self):
        """continue from checkpoint"""
        print("Resuming from index", max(self.response_hash.keys()))

        for index in tqdm(range(max(self.response_hash.keys()), max(self.response_hash.keys()) + 75000)):

            try:
                if index % 1000 == 0:
                    print("Creating checkpoint in version", self.ckpt_version + 1)
                    self.store_self_checkpoint()

                if not index in self.response_hash.keys():

                    response = self.get_page_response(index)
                    if response is not None:
                        store_path = self.make_dir_to_store(index)
                        self.store_page_response(response, store_path, index)
                        self.response_hash[index] = "200"
                        time.sleep(self.TIME_DELAY)
                    else:
                        if self.verbose:
                            print("None found for index", index)
                            time.sleep(0.5)
                elif self.verbose:
                    print("already checked index:", index)

            except Exception as e:
                print("ERROR for index", index)
                print(e)


def store_checkpoint(class_object):
    """explicitly store checkpoint"""
    class_object.ckpt_version += 1
    pickle.dump(class_object,
                open(os.path.join(class_object.STORAGE, "anon_ckpt_" + str(class_object.ckpt_version) + ".pkl"),
                     'wb'))
    pass


def load_specific_checkpoint(pathlike):
    """continue from specific checkpoint"""
    return pickle.load(open(pathlike, 'rb'))


def load_latest_class_checkpoint(komoot_dir):
    """get the latest checkpoint from storage-dir, continue"""
    list_of_files = glob.glob(os.path.join(komoot_dir, "*.pkl"))
    latest_ckpt = max(list_of_files, key=os.path.getctime)
    return pickle.load(open(latest_ckpt, 'rb'))


def get_parser(**parser_kwargs):
    """parse arguments from command-line"""
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-s",
        "--storage_dir",
        type=str,
        help="to store the data",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    # could later add a logic here to to do some args parsing for now, no
    komoot = load_latest_class_checkpoint(args.storage_dir)
    komoot.say_hello()
    komoot.continue_index_search()
    komoot.store_self_checkpoint()
    print("Ended successfully.")