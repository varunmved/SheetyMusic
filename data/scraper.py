import requests
import re
import wget
import pdb



url = "http://www.mutopiaproject.org/cgibin/make-table.cgi?searchingfor=&Composer=&Instrument=Piano&\
Style=Classical&solo=1&timelength=1&timeunit=week"


offset = 0
while True:
    print("Downloading from startat = %s" % offset)
    params = {"startat": offset}

    r = requests.get(url, params=params)

    # all the html elements that contain a download url for a lyfile
    lyfiles = [line for line in r.text.split("\n") if "Download" in line]

    if len(lyfiles) == 0: break

    for lyfile in lyfiles:
        start = lyfile.index("href=") + 6
        try:
            end = lyfile.index(".ly\"") + 3
            if end > start:
                wget.download(lyfile[start:end], "classical-raw/")

        except ValueError:
            continue
    
    offset += 10
