import urllib2
from time import sleep

from bs4 import BeautifulSoup
from selenium import webdriver


machine_folk_url_base = "https://themachinefolksession.org/tune/"
MAX_MACHINE_FOLK_TUNES = 560
machine_folk_output_path = 'data/machine_samples_full.txt'
generator_folk_url_base = "https://folkrnn.org/tune/"
MAX_GENERATOR_FOLK_TUNES = 23020
generator_output_path = 'data/generator_samples_full.txt'

acceptable_scales = set()

with open('data/tunes.txt', 'r') as f:
    for line in f:
        if line.startswith("K:"):
            scale = line.strip()
            if scale.lower() not in acceptable_scales:
                acceptable_scales.add(scale.lower())
                print("Acceptable scale: {}".format(scale))


def get_tunes(url, last_tune, num_tunes, file_path, needs_loading=False, box_id_needs_index=False):
    driver = webdriver.Chrome()
    with open(file_path, 'w+') as f:
        for i in range(last_tune, last_tune-num_tunes, -1):
            if needs_loading:
                driver.get("{}{}".format(url, i))
                page = driver.page_source
            else:
                page = urllib2.urlopen("{}{}".format(url, i))
            sleep(2)
            soup = BeautifulSoup(page, 'html.parser')
            box_id = "abc-{}".format(i) if box_id_needs_index else "abc-tune"
            name_box = soup.find(id=box_id)
            #test_box = driver.find_element_by_id(box_id)
            #name_box = test_box.text
            if name_box is not None:
                if len(name_box.text.strip()) > 0:
                    tune = name_box.text.strip()
                    print("Scraped tune {}".format(i))
                    write_data_from_tune(tune, f)


def write_data_from_tune(tune, f):
    tunearr = tune.split("\n")
    m = None
    k = None
    q = None
    l = None
    v = None
    x = None
    abc = []
    for line in tunearr:
        if line.startswith("M:"):
            m = line
        elif line.startswith("K:"):
            k = line
        elif line.startswith("Q:"):
            q = line
        elif line.startswith("L:"):
            l = line
        elif line.startswith("V:"):
            v = line
        elif line.startswith("X:"):
            x = line
        else:
            abc.append(line)
    if m is not None and k is not None and len(abc) > 0:
        k = "{}".format(k).replace(" ", "")
        if k.lower() in acceptable_scales:
            m = "{}".format(m).replace(" ", "")
            abc = ''.join(abc)
            abc = ''.join(abc.split())
            abc = ' '.join(abc)
            try:
                f.write("{}\n{}\n{}\n\n".format(m, k, abc))
            except UnicodeEncodeError:
                pass


# get_tunes(generator_folk_url_base,
#           MAX_GENERATOR_FOLK_TUNES,
#           500,
#           generator_output_path,
#           needs_loading=True,
#           box_id_needs_index=True)
get_tunes(machine_folk_url_base,
          MAX_MACHINE_FOLK_TUNES,
          500,
          machine_folk_output_path)
