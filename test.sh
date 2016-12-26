#!/bin/bash

b=(a b c d e f g h i j)

b[0]='/home/work/data/twitter_crawl_daily/tweets-20160120-json.txt.gz'

b[1]='/home/work/data/twitter_crawl_daily/tweets-20160220-json.txt.gz'

b[2]='/home/work/data/twitter_crawl_daily/tweets-20160321-json.txt.gz'

b[3]='/home/work/data/twitter_crawl_daily/tweets-20160420-json.txt.gz'

b[4]='/home/work/data/twitter_crawl_daily/tweets-20160520-json.txt.gz'

b[5]='/home/work/data/twitter_crawl_daily/tweets-20160620-json.txt.gz'

b[6]='/home/work/data/twitter_crawl_daily/tweets-20160720-json.txt.gz'

b[7]='/home/work/data/twitter_crawl_daily/tweets-20160820-json.txt.gz'

b[8]='/home/work/data/twitter_crawl_daily/tweets-20160920-json.txt.gz'

b[9]='/home/work/data/twitter_crawl_daily/tweets-20161020-json.txt.gz'

b[10]='/home/work/data/twitter_crawl_daily/tweets-20161120-json.txt.gz'

b[11]='/home/work/data/twitter_crawl_daily/tweets-20161220-json.txt.gz'

b[12]='/home/work/data/twitter_crawl_daily/tweets-20160110-json.txt.gz'

b[13]='/home/work/data/twitter_crawl_daily/tweets-20160210-json.txt.gz'

b[14]='/home/work/data/twitter_crawl_daily/tweets-20160310-json.txt.gz'

b[15]='/home/work/data/twitter_crawl_daily/tweets-20160410-json.txt.gz'

b[16]='/home/work/data/twitter_crawl_daily/tweets-20160510-json.txt.gz'

b[17]='/home/work/data/twitter_crawl_daily/tweets-20160610-json.txt.gz'

b[18]='/home/work/data/twitter_crawl_daily/tweets-20160710-json.txt.gz'

b[19]='/home/work/data/twitter_crawl_daily/tweets-20160810-json.txt.gz'

b[20]='/home/work/data/twitter_crawl_daily/tweets-20160910-json.txt.gz'

b[21]='/home/work/data/twitter_crawl_daily/tweets-20161010-json.txt.gz'

b[22]='/home/work/data/twitter_crawl_daily/tweets-20161110-json.txt.gz'

b[23]='/home/work/data/twitter_crawl_daily/tweets-20161211-json.txt.gz'

b[24]='/home/work/data/twitter_crawl_daily/tweets-20160131-json.txt.gz'

b[25]='/home/work/data/twitter_crawl_daily/tweets-20160229-json.txt.gz'

b[26]='/home/work/data/twitter_crawl_daily/tweets-20160330-json.txt.gz'

b[27]='/home/work/data/twitter_crawl_daily/tweets-20160430-json.txt.gz'

b[28]='/home/work/data/twitter_crawl_daily/tweets-20160530-json.txt.gz'

b[29]='/home/work/data/twitter_crawl_daily/tweets-20160630-json.txt.gz'

b[30]='/home/work/data/twitter_crawl_daily/tweets-20160730-json.txt.gz'

b[31]='/home/work/data/twitter_crawl_daily/tweets-20160830-json.txt.gz'

b[32]='/home/work/data/twitter_crawl_daily/tweets-20160930-json.txt.gz'

b[33]='/home/work/data/twitter_crawl_daily/tweets-20161030-json.txt.gz'

b[34]='/home/work/data/twitter_crawl_daily/tweets-20161130-json.txt.gz'

echo -n >corpus.json

for var in {0..34}
	   
do
    echo "epoc${var}"
    echo -n >newfile${var}.json
    python ~/souzoukougaku/create_corpus.py ${b[$var]} >newfile{$var}    
    cat newfile{$var} >>corpus.json
done

python ~/souzoukougaku/create_dicts.py corpus.json

python ~/souzoukougaku/create_data.py  corpus.json dicts.pkl
