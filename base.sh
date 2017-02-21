cd py

i=0
for lang in eng tur ara; do
    for top_affix in 100 200 300 400 500; do
        for top_words in 2500 5000 10000; do
            for compounding in 0 1 ; do
                echo $lang $top_affix $top_words $compounding 0;
                file=$lang-${top_affix}-$top_words-$compounding.results
                if [ -f base/$file ]; then
                    echo file exists;
                    continue;
                fi
                i=$((i+1));
                if [ $((i % 5)) -gt 0 ]; then
                    python run.py $lang $top_affix $top_words $compounding 0 > base/$file &
                else
                    python run.py $lang $top_affix $top_words $compounding 0 > base/$file ;
                fi
                sleep 30
            done
        done
    done
done
