from google_images_download import google_images_download
import cv2
import os
from random import random
import shutil

response = google_images_download.googleimagesdownload()

# search_queries = ['BMW for sale', 'Lexus for sale', 'Alfa romeo for sale']  # 'tibetan mastiff','eurasier'
# search_queries = [ 'samochod', 'quad']  # 'tibetan mastiff','eurasier'
# search_queries = ['mclaren track']
search_queries = ['motocykl']

def downloadimages(query):
    arguments = {"keywords": query,
                 "format": "jpg",
                 "limit": 550,
                 "print_urls": True,
                 "chromedriver": "chromedriver.exe"
                 }
    try:
        response.download(arguments)

        # Handling File NotFound Error
    except FileNotFoundError:
        arguments = {"keywords": query,
                     "format": "jpg",
                     "limit": 4,
                     "print_urls": True,
                     "size": "medium",
                     "chromedriver": "chromedriver.exe"}

        # Providing arguments for the searched query
        try:
            # Downloading the photos based
            # on the given arguments
            response.download(arguments)
        except:
            pass


def transform_pictures():
    # save_in = 'prepared'
    # oryginal_in = 'downloads'
    for query in search_queries:
        rootDir = 'downloads/' + query
        saveDir = 'prepared/' + query
        for dirName, subdirList, fileList in os.walk(rootDir):
            for fname in fileList:
                try:
                    tmp = rootDir + '/' + fname
                    print(tmp)
                    im = cv2.imread(tmp)
                    im_n = cv2.resize(im, (32, 32))
                    cv2.imwrite(saveDir + '/' + fname, im_n)
                except Exception:
                    continue


def split_data():
    training_percent = 0.7
    for query in search_queries:
        root_dir = 'prepared/' + query
        for dirName, subdirList, fileList in os.walk(root_dir):
            for fname in fileList:
                try:
                    source = root_dir + '/' + fname
                    print(source)

                    if random() < training_percent:
                        destiny = 'training/' + query + '/' + fname
                    else:
                        destiny = 'validation/' + query + '/' + fname
                    shutil.copy(source, destiny)
                except Exception:
                    continue


""" Download and preproccess"""
for query in search_queries:
    downloadimages(query)
    print()

transform_pictures()
# split_data()
